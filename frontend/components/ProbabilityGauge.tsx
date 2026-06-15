export function ProbabilityGauge({
  probability,
  interval,
}: {
  probability: number;
  interval: [number, number];
}) {
  const pct = Math.round(probability * 100);
  const lo = Math.round(interval[0] * 100);
  const hi = Math.round(interval[1] * 100);
  const color =
    probability < 0.4 ? "#1a7f37" : probability < 0.6 ? "#9a6700" : "#cf222e";
  return (
    <div className="w-full">
      <div className="mb-1 flex justify-between text-sm text-gray-400">
        <span>Deepfake probability</span>
        <span className="font-mono">{pct}%</span>
      </div>
      <div className="relative h-4 w-full overflow-hidden rounded-full bg-gray-800">
        {/* Confidence interval band */}
        <div
          className="absolute h-full bg-gray-600/50"
          style={{ left: `${lo}%`, width: `${Math.max(hi - lo, 1)}%` }}
        />
        {/* Point estimate */}
        <div
          className="absolute h-full"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <div className="mt-1 text-xs text-gray-500">
        95% CI: {lo}% – {hi}%
      </div>
    </div>
  );
}
