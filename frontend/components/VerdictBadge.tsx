import type { Verdict } from "@/lib/api";

const STYLES: Record<Verdict, string> = {
  AUTHENTIC: "bg-green-900 text-green-200 border-green-600",
  LEANING_AUTHENTIC: "bg-green-950 text-green-300 border-green-700",
  UNCERTAIN: "bg-yellow-900 text-yellow-200 border-yellow-600",
  LIKELY_SYNTHETIC: "bg-orange-900 text-orange-200 border-orange-600",
  HIGH_CONFIDENCE_SYNTHETIC: "bg-red-900 text-red-200 border-red-600",
};

export function VerdictBadge({ verdict }: { verdict: Verdict }) {
  return (
    <span
      className={`inline-block rounded-md border px-3 py-1 text-sm font-semibold ${STYLES[verdict]}`}
    >
      {verdict.replaceAll("_", " ")}
    </span>
  );
}
