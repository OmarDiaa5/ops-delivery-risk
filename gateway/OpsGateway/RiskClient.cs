using System.Net;
using System.Net.Http.Json;
using System.Text.Json.Serialization;

namespace OpsGateway;

public record RiskResponse(
    [property: JsonPropertyName("order_id")] string? OrderId,
    [property: JsonPropertyName("late_risk")] double LateRisk,
    [property: JsonPropertyName("flagged")] bool Flagged,
    [property: JsonPropertyName("tier")] string Tier,
    [property: JsonPropertyName("top_drivers")] List<string> TopDrivers);

public record ReviewQueueRequest(List<string> OrderIds);

/// <summary>Typed client for the Python late-delivery risk service.</summary>
public class RiskClient(HttpClient http)
{
    public async Task<object?> HealthAsync()
    {
        try { return await http.GetFromJsonAsync<object>("/health"); }
        catch (HttpRequestException) { return null; }
    }

    public async Task<RiskResponse?> OrderRiskAsync(string orderId)
    {
        var resp = await http.GetAsync($"/orders/{Uri.EscapeDataString(orderId)}/risk");
        if (resp.StatusCode == HttpStatusCode.NotFound) return null;
        resp.EnsureSuccessStatusCode();
        return await resp.Content.ReadFromJsonAsync<RiskResponse>();
    }

    public async Task<string?> DriftAsync(string? month)
    {
        var url = string.IsNullOrEmpty(month) ? "/monitoring/drift" : $"/monitoring/drift?month={Uri.EscapeDataString(month)}";
        var resp = await http.GetAsync(url);
        if (resp.StatusCode == HttpStatusCode.NotFound) return null;
        resp.EnsureSuccessStatusCode();
        return await resp.Content.ReadAsStringAsync();
    }
}
