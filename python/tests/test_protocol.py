import io
import struct
import unittest

from python.communication.observe import receive


class ProtocolTest(unittest.TestCase):
    def test_status(self):
        meta = b'{"ok":true,"scene":"menu"}'
        data = struct.pack("<4sII", b"NFL1", len(meta), 0) + meta
        self.assertEqual(receive(io.BytesIO(data), False), ({"ok": True, "scene": "menu"}, b""))

    def test_refuses_unbounded_response(self):
        data = struct.pack("<4sII", b"NFL1", 10, 2**31)
        with self.assertRaises(ValueError):
            receive(io.BytesIO(data), True)

    def test_short_payload_is_not_an_observation(self):
        data = struct.pack("<4sII", b"NFL1", 11, 5) + b'{"ok":true}xy'
        with self.assertRaises(EOFError):
            receive(io.BytesIO(data), True)
