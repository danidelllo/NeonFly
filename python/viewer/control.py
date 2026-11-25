"""Shared action-boundary controls. Only the model worker talks to Unity."""
import multiprocessing as mp
import time


class TrainingControl:
    phases = ('starting', 'playing', 'paused', 'stepping', 'stopped')

    def __init__(self):
        self.condition = mp.Condition()
        self.paused = mp.Value('b', False, lock=False)
        self.credits = mp.Value('i', 0, lock=False)
        self.phase = mp.Value('i', 0, lock=False)
        self.actions = mp.Value('q', 0, lock=False)

    def snapshot(self):
        with self.condition:
            return dict(state=self.phases[self.phase.value], pauseRequested=bool(self.paused.value),
                        completedActions=self.actions.value)

    def command(self, command):
        with self.condition:
            if self.phase.value in (0, 4):
                raise ValueError('Controller is not ready')
            if command == 'step':
                if not self.paused.value or self.phase.value != 2 or self.credits.value:
                    raise ValueError('Pause before stepping')
                self.credits.value = 1
                self.phase.value = 3
            elif command in ('pause', 'play'):
                self.paused.value = command == 'pause'
                self.credits.value = 0
            else:
                raise ValueError('Unknown control command')
            self.condition.notify_all()
        return self.snapshot()

    def wait(self, stop, heartbeat, *, consume=False):
        """Pause before action/reset; a reset does not consume a step credit."""
        last_heartbeat = 0.
        while not stop.is_set():
            with self.condition:
                if not self.paused.value or self.credits.value:
                    self.phase.value = 3 if self.paused.value else 1
                    if consume and self.paused.value:
                        self.credits.value -= 1
                    return True
                self.phase.value = 2
            # Renew the bridge hold on the worker's own connection, preventing
            # its abandoned-controller watchdog from unfreezing a long pause.
            if time.monotonic() - last_heartbeat >= 1:
                heartbeat()
                last_heartbeat = time.monotonic()
            with self.condition:
                self.condition.wait(timeout=.1)
        return False

    def completed(self):
        with self.condition:
            self.actions.value += 1

    def close(self):
        with self.condition:
            self.phase.value = 4
            self.condition.notify_all()
