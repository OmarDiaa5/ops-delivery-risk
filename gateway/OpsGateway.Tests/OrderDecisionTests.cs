using System.Net;
using System.Text;
using OpsGateway;
using Xunit;

public class OrderDecisionTests
{
    static RiskResponse R(string tier, params string[] drivers) =>
        new("o1", 0.8, tier == "high", tier, drivers.ToList());

    [Fact]
    public void HighRiskFromSellerHistory_ContactsSeller() =>
        Assert.StartsWith("Contact the seller", OrderDecision.ActionFor(R("high", "seller_hist_late_rate_high")));

    [Fact]
    public void HighRiskOtherwise_UpgradesCarrier() =>
        Assert.StartsWith("Upgrade the carrier", OrderDecision.ActionFor(R("high", "distance_km_high")));

    [Theory]
    [InlineData("medium", "Watch")]
    [InlineData("low", "No action")]
    public void LowerTiers(string tier, string expected) =>
        Assert.StartsWith(expected, OrderDecision.ActionFor(R(tier)));
}

public class RiskClientTests
{
    class StubHandler(HttpStatusCode code, string body) : HttpMessageHandler
    {
        public string? LastPath;
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage req, CancellationToken ct)
        {
            LastPath = req.RequestUri!.PathAndQuery;
            return Task.FromResult(new HttpResponseMessage(code)
            { Content = new StringContent(body, Encoding.UTF8, "application/json") });
        }
    }

    static RiskClient Client(StubHandler h) => new(new HttpClient(h) { BaseAddress = new Uri("http://risk") });

    [Fact]
    public async Task ParsesSnakeCaseResponse()
    {
        var h = new StubHandler(HttpStatusCode.OK,
            """{"order_id":"abc","late_risk":0.91,"flagged":true,"tier":"high","top_drivers":["promised_days"]}""");
        var r = await Client(h).OrderRiskAsync("abc");
        Assert.NotNull(r);
        Assert.Equal(0.91, r!.LateRisk);
        Assert.True(r.Flagged);
        Assert.Equal("/orders/abc/risk", h.LastPath);
    }

    [Fact]
    public async Task UnknownOrder_ReturnsNull()
    {
        var r = await Client(new StubHandler(HttpStatusCode.NotFound, "{}")).OrderRiskAsync("missing");
        Assert.Null(r);
    }
}
