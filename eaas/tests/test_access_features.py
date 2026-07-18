import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import eaas.app as app_module
import eaas.db as db_module
import eaas.ml_core as ml_core_module


class AccessFeatureTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "test_eaas.db")
        db_module.DB_PATH = self.db_path
        app_module.DB_PATH = self.db_path
        db_module.init_db()

    def tearDown(self):
        self.tmpdir.cleanup()

    @patch.object(app_module, "retrain_face_recognizer")
    @patch.object(app_module.face_recognizer, "predict")
    def test_find_existing_face_match_returns_existing_user(self, mock_predict, mock_retrain):
        conn = db_module.get_conn()
        cur = conn.execute(
            "INSERT INTO users (full_name, matric_no) VALUES (?, ?)",
            ("Ada", "202300001"),
        )
        conn.commit()

        mock_predict.return_value = (cur.lastrowid, 97.5)
        app_module.face_recognizer.trained = False
        mock_retrain.side_effect = lambda conn: setattr(app_module.face_recognizer, "trained", True)

        try:
            matched_user_id = app_module.find_existing_face_match(
                conn,
                [np.zeros((160, 160), dtype=np.uint8)],
            )
        finally:
            conn.close()

        self.assertEqual(matched_user_id, cur.lastrowid)
        mock_retrain.assert_called_once()

    def test_delete_access_log_removes_row(self):
        conn = db_module.get_conn()
        cur = conn.execute(
            "INSERT INTO access_logs (attempt_name, decision) VALUES (?, ?)",
            ("Test User", "GRANTED"),
        )
        conn.commit()

        app_module.delete_access_log(conn, cur.lastrowid)
        conn.commit()

        row = conn.execute(
            "SELECT id FROM access_logs WHERE id = ?",
            (cur.lastrowid,),
        ).fetchone()
        self.assertIsNone(row)
        conn.close()

    def test_delete_user_removes_user_and_related_files(self):
        conn = db_module.get_conn()
        user_id = conn.execute(
            "INSERT INTO users (full_name, matric_no, photo_path) VALUES (?, ?, ?)",
            ("Grace", "202300002", "demo.jpg"),
        ).lastrowid
        conn.execute(
            "INSERT INTO access_logs (user_id, attempt_name, decision, image_path) VALUES (?, ?, ?, ?)",
            (user_id, "Grace", "GRANTED", "demo.jpg"),
        )
        conn.commit()

        with open(os.path.join(self.tmpdir.name, "demo.jpg"), "w") as fh:
            fh.write("sample")

        app_module.delete_user(conn, user_id)
        conn.commit()

        user_row = conn.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone()
        log_row = conn.execute("SELECT id FROM access_logs WHERE user_id = ?", (user_id,)).fetchone()
        self.assertIsNone(user_row)
        self.assertIsNone(log_row)
        conn.close()

    def test_decide_access_requires_high_emotion_confidence_for_success(self):
        decision, reason, level = ml_core_module.decide_access(
            identity_conf=80.0,
            emotion_label="Happy",
            emotion_conf=85.0,
            baseline_emotion="Neutral",
            face_detected=True,
        )
        self.assertEqual(decision, "SUCCESS")
        self.assertEqual(level, "success")
        self.assertIn("Detected emotion: Happy (85.0%)", reason)

    def test_decide_access_warns_when_emotion_confidence_is_low(self):
        decision, reason, level = ml_core_module.decide_access(
            identity_conf=80.0,
            emotion_label="Happy",
            emotion_conf=60.0,
            baseline_emotion="Neutral",
            face_detected=True,
        )
        self.assertEqual(decision, "ADDITIONAL VERIFICATION REQUIRED")
        self.assertEqual(level, "warning")
        self.assertIn("emotion confidence is lower than 80.0%", reason)

    def test_decide_access_denies_when_both_confidences_are_low(self):
        decision, reason, level = ml_core_module.decide_access(
            identity_conf=20.0,
            emotion_label="Happy",
            emotion_conf=50.0,
            baseline_emotion="Neutral",
            face_detected=True,
        )
        self.assertEqual(decision, "DENIED")
        self.assertEqual(level, "danger")
        self.assertIn("both emotion confidence (50.0%) and identity confidence (20.0%) are insufficient", reason)


if __name__ == "__main__":
    unittest.main()
