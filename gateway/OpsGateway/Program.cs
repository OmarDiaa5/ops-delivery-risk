using OpsGateway;

var builder = WebApplication.CreateBuilder(args);

var riskApiUrl = builder.Configuration["RISK_API_URL"] ?? "http://localhost:8000";
builder.Services.AddHttpClient<RiskClient>(c =>
{
    c.BaseAddress = new Uri(riskApiUrl);
    c.Timeout = TimeSpan.FromSeconds(10);
});

var app = builder.Build();
app.UseDefaultFiles();
app.UseStaticFiles();

app.MapGet("/api/health", async (RiskClient risk) =>
{
    var upstream = await risk.HealthAsync();
    return upstream is null
        ? Results.Problem("risk service unreachable", statusCode: 503)
        : Results.Ok(new { gateway = "ok", riskService = upstream });
});

// One order: the model's score plus the action the ops team should take.
app.MapGet("/api/orders/{orderId}/risk", async (string orderId, RiskClient risk) =>
{
    var r = await risk.OrderRiskAsync(orderId);
    return r is null ? Results.NotFound() : Results.Ok(OrderDecision.From(r));
});

// A shift's worth of orders, riskiest first: the ops review queue.
app.MapPost("/api/orders/review-queue", async (ReviewQueueRequest req, RiskClient risk) =>
{
    if (req.OrderIds is null || req.OrderIds.Count == 0 || req.OrderIds.Count > 500)
        return Results.BadRequest("send 1-500 order ids");

    var tasks = req.OrderIds.Distinct().Select(risk.OrderRiskAsync);
    var scored = (await Task.WhenAll(tasks)).Where(r => r is not null).Select(r => OrderDecision.From(r!));
    var queue = scored.OrderByDescending(d => d.LateRisk).ToList();
    return Results.Ok(new
    {
        total = queue.Count,
        flagged = queue.Count(d => d.Flagged),
        orders = queue,
    });
});

app.MapGet("/api/monitoring/drift", async (string? month, RiskClient risk) =>
{
    var json = await risk.DriftAsync(month);
    return json is null ? Results.NotFound() : Results.Content(json, "application/json");
});

app.Run();

public partial class Program;
