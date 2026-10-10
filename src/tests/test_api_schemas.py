import unittest

from pydantic import ValidationError

from src.api.schemas import AnalyzeRequest
from src.api.ui import INDEX_HTML


class SchemaTests(unittest.TestCase):
    def test_subject_and_body_are_joined_like_the_dataset(self):
        self.assertEqual(AnalyzeRequest(subject="Reset your password.", body="Use this link.").email_text(),
                         "Reset your password. Use this link.")

    def test_text_wins_and_is_stripped(self):
        self.assertEqual(AnalyzeRequest(text="  hello  ", subject="x", body="y").email_text(), "hello")

    def test_body_only_or_subject_only(self):
        self.assertEqual(AnalyzeRequest(body="just a body").email_text(), "just a body")
        self.assertEqual(AnalyzeRequest(subject="just a subject").email_text(), "just a subject")

    def test_empty_is_rejected(self):
        for kwargs in [{}, {"text": "   "}, {"subject": " ", "body": " "}]:
            with self.assertRaises(ValidationError):
                AnalyzeRequest(**kwargs)

    def test_too_long_is_rejected(self):
        with self.assertRaises(ValidationError):
            AnalyzeRequest(text="a" * 20001)


class UiTests(unittest.TestCase):
    def test_page_is_embedded_in_code(self):
        self.assertIn("PhishGuard AI", INDEX_HTML)
        self.assertIn("/analyze", INDEX_HTML)
        self.assertTrue(INDEX_HTML.lstrip().lower().startswith("<!doctype html>"))


if __name__ == "__main__":
    unittest.main()
