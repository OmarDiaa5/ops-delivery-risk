namespace OpsGateway;

/// <summary>Turns a model score into what the operations team does next.</summary>
public record OrderDecision(string? OrderId, double LateRisk, bool Flagged, string Tier,
                            List<string> TopDrivers, string RecommendedAction)
{
    public static OrderDecision From(RiskResponse r) =>
        new(r.OrderId, r.LateRisk, r.Flagged, r.Tier, r.TopDrivers, ActionFor(r));

    public static string ActionFor(RiskResponse r) => r.Tier switch
    {
        "high" when r.TopDrivers.Contains("seller_hist_late_rate_high")
            => "Contact the seller to confirm dispatch today; notify the customer of a possible delay",
        "high" => "Upgrade the carrier or reroute; notify the customer of a possible delay",
        "medium" => "Watch: check tracking at the halfway point of the promised window",
        _ => "No action",
    };
}
