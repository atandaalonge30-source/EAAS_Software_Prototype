import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import app as app_module
import db as db_module


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


if __name__ == "__main__":
    unittest.main()
