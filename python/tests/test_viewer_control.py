import threading
import time
import unittest

from python.viewer.control import TrainingControl


class ControlTests(unittest.TestCase):
    def ready(self):
        control = TrainingControl()
        self.assertTrue(control.wait(threading.Event(), lambda: None))
        return control

    def test_step_requires_pause_and_grants_exactly_one_action(self):
        control = self.ready()
        with self.assertRaises(ValueError): control.command('step')
        control.command('pause')
        stop = threading.Event(); advanced = []
        def worker():
            while control.wait(stop, lambda: None, consume=True):
                advanced.append(1); control.completed()
        thread = threading.Thread(target=worker); thread.start()
        self.wait_state(control, 'paused')
        self.assertEqual(advanced, [])
        for expected in (1, 2):
            control.command('step')
            self.wait_state(control, 'paused')
            self.assertEqual(len(advanced), expected)
            self.assertEqual(control.snapshot()['completedActions'], expected)
        stop.set(); thread.join(2)
        self.assertFalse(thread.is_alive())

    def wait_state(self, control, state):
        end = time.monotonic()+2
        while control.snapshot()['state'] != state and time.monotonic() < end:
            time.sleep(.005)
        self.assertEqual(control.snapshot()['state'], state)

    def test_pause_heartbeat_and_play_unblock(self):
        control = self.ready(); control.command('pause')
        stop = threading.Event(); heartbeats = []; permitted = []
        thread = threading.Thread(target=lambda: permitted.append(control.wait(stop, lambda: heartbeats.append(1))))
        thread.start(); self.wait_state(control, 'paused')
        control.command('play'); thread.join(2)
        self.assertEqual(permitted, [True]); self.assertTrue(heartbeats)

    def test_reset_does_not_consume_step_credit(self):
        control = self.ready(); control.command('pause'); control.phase.value = 2
        control.command('step'); stop = threading.Event()
        self.assertTrue(control.wait(stop, lambda: None))
        self.assertEqual(control.credits.value, 1)
        self.assertTrue(control.wait(stop, lambda: None, consume=True))
        self.assertEqual(control.credits.value, 0)
        with self.assertRaises(ValueError): control.command('step')
        control.close()
        with self.assertRaises(ValueError): control.command('play')

    def test_unknown_commands_are_rejected(self):
        control = self.ready()
        with self.assertRaises(ValueError): control.command('reset')

    def test_terminal_preview_handles_lethal_damage_and_removed_ship(self):
        from python.viewer.live import terminal_resources
        previous = [.5, .2, -.8]
        values = terminal_resources(dict(health=-959, maxHealth=40,
            bullets=60, maxBullets=100, rockets=0, maxRockets=10), previous)
        for actual, expected in zip(values, [-1., .2, -1.]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(terminal_resources(dict(health=0, maxHealth=-1,
            bullets=-1, maxBullets=-1, rockets=-1, maxRockets=-1), previous), previous)
