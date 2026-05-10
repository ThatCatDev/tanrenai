import type { Activity } from '../state';

interface Props {
  activity: Activity;
  iteration: number;
  maxIterations: number;
}

export function ActivityBar({ activity, iteration, maxIterations }: Props) {
  if (activity.kind === 'idle') {
    return null;
  }
  const label = labelFor(activity);
  const iter =
    iteration > 0
      ? maxIterations > 0
        ? ` · iteration ${iteration}/${maxIterations}`
        : ` · iteration ${iteration}`
      : '';

  return (
    <div class="activity">
      <span class="activity-dot" />
      <span class="activity-label">
        {label}
        {iter}
      </span>
    </div>
  );
}

function labelFor(activity: Activity): string {
  switch (activity.kind) {
    case 'thinking':
      return 'thinking…';
    case 'generating':
      return 'generating…';
    case 'preparing':
      return `preparing ${activity.name}… (${formatChars(activity.chars)})`;
    case 'tool':
      return `running ${activity.name}…`;
    case 'awaiting_approval':
      return `awaiting your approval to run ${activity.name}`;
    default:
      return '';
  }
}

function formatChars(n: number): string {
  if (n >= 1000) {
    return `${(n / 1000).toFixed(1)}k chars`;
  }

  return `${n} chars`;
}
