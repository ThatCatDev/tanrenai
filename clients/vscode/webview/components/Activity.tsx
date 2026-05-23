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

  return (
    <div class="activity">
      <span class="activity-dashes" aria-hidden="true" />
      <span class="activity-label">
        {labelFor(activity)}
        {iteration > 0 && (
          <span class="activity-iter">
            {' · '}
            {maxIterations > 0 ? `iter ${iteration}/${maxIterations}` : `iter ${iteration}`}
          </span>
        )}
      </span>
    </div>
  );
}

function labelFor(activity: Activity): string {
  switch (activity.kind) {
    case 'thinking':
      return 'thinking';
    case 'generating':
      return 'generating';
    case 'preparing':
      return `preparing ${activity.name} (${formatChars(activity.chars)})`;
    case 'tool':
      return `running ${activity.name}`;
    case 'awaiting_approval':
      return `awaiting approval — ${activity.name}`;
    default:
      return '';
  }
}

function formatChars(n: number): string {
  if (n >= 1000) {
    return `${(n / 1000).toFixed(1)}k`;
  }

  return `${n}`;
}
