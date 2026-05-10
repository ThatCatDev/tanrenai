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
    case 'tool':
      return `running ${activity.name}…`;
    default:
      return '';
  }
}
