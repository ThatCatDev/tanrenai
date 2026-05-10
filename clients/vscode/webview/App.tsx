import { useEffect, useReducer } from 'preact/hooks';
import { Composer } from './components/Composer';
import { Header } from './components/Header';
import { MessageList } from './components/MessageList';
import { StatusPanel } from './components/StatusPanel';
import { onMessage } from './host';
import { initialState, reduce } from './state';

export function App() {
  const [state, dispatch] = useReducer(reduce, initialState);

  useEffect(() => {
    return onMessage(dispatch);
  }, []);

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
