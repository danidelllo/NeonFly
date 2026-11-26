import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from python.operations.combat_handoff import finished, check_manifest, quote


class HandoffTests(unittest.TestCase):
    def test_waits_for_full_budget_and_final_evaluation(self):
        self.assertFalse(finished([], []))
        updates = [dict(update=128, episodes=512)]
        self.assertFalse(finished(updates, [dict(update=112, episode=448)]))
        self.assertTrue(finished(updates, [dict(update=128, episode=512)]))
        self.assertFalse(finished([dict(update=127, episodes=508)], [dict(update=128, episode=512)]))

    def test_modified_release_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); file = root/'bridge.dll'; file.write_bytes(b'audited')
            manifest = dict(files={'bridge.dll': hashlib.sha256(file.read_bytes()).hexdigest()})
            with patch('python.operations.combat_handoff.ROOT', root):
                check_manifest(manifest)
                file.write_bytes(b'changed')
                with self.assertRaisesRegex(RuntimeError, 'changed'):
                    check_manifest(manifest)

    def test_powershell_paths_are_literal(self):
        self.assertEqual(quote("C:/a'b"), "'C:/a''b'")
