import { useEffect, useReducer } from 'preact/hooks';
import { Composer } from './components/Composer';
import { Header } from './components/Header';
import { MessageList } from './components/MessageList';
import { StatusPanel } from './components/StatusPanel';
import { getPersistedShell, onMessage, setPersistedShell } from './host';
import { initialState, reduce, type AppState } from './state';

export function App() {
  const [state, dispatch] = useReducer(reduce, initialState, init);

  useEffect(() => {
    return onMessage(dispatch);
  }, []);

  // Persist visual shell (connection + mode) so a remount paints the chat
  // immediately instead of flashing the idle panel. Entries are not
  // persisted — the controller's transcript replays them.
  useEffect(() => {
    setPersistedShell({ connection: state.connection, mode: state.mode });
  }, [state.connection, state.mode]);

  return renderRoot(state);
}

function init(seed: AppState): AppState {
  const persisted = getPersistedShell();
  if (!persisted) {
    return seed;
  }

  return { ...seed, connection: persisted.connection, mode: persisted.mode };
}

function renderRoot(state: AppState) {
  if (state.connection.status !== 'connected') {
    return (
      <div class="root">
        <StatusPanel connection={state.connection} />
      </div>
    );
  }

  return (
    <div class="root">
      <Header
        model={state.connection.model}
        toolCount={state.connection.toolCount}
        mode={state.mode}
      />
      <MessageList entries={state.entries} />
      <Composer turnRunning={state.turnRunning} mode={state.mode} />
    </div>
  );
}
