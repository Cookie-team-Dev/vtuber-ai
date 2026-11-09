import io
import sys
import tempfile
import types
import unittest
import wave
from unittest import mock

import voice


class AmikaTTSClientTests(unittest.TestCase):
    @staticmethod
    def _wav_bytes(frames):
        output = io.BytesIO()
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            wav_file.writeframes(frames)
        return output.getvalue()

    def test_voice_chat_synthesis_prefers_amika(self):
        with (
            mock.patch.object(voice, "_synthesize_with_amika") as amika,
            mock.patch.object(voice, "_synthesize_with_piper") as piper,
        ):
            voice.synthesize_for_voice_chat("hello", "/tmp/voice-chat.wav")

        amika.assert_called_once_with("hello", "neutral", "/tmp/voice-chat.wav")
        piper.assert_not_called()

    def test_voice_chat_synthesis_falls_back_to_piper(self):
        with (
            mock.patch.object(
                voice, "_synthesize_with_amika", side_effect=RuntimeError("offline")
            ),
            mock.patch.object(voice, "_synthesize_with_piper") as piper,
        ):
            voice.synthesize_for_voice_chat("hello", "/tmp/voice-chat.wav")

        piper.assert_called_once_with("hello", "/tmp/voice-chat.wav", 1.0)

    def test_voice_chat_synthesis_uses_piper_cli_when_library_fails(self):
        with (
            mock.patch.object(
                voice, "_synthesize_with_amika", side_effect=RuntimeError("offline")
            ),
            mock.patch.object(
                voice, "_synthesize_with_piper", side_effect=RuntimeError("unavailable")
            ),
            mock.patch.object(voice, "_synthesize_with_cli") as piper_cli,
        ):
            voice.synthesize_for_voice_chat("hello", "/tmp/voice-chat.wav")

        piper_cli.assert_called_once_with("hello", "/tmp/voice-chat.wav", 1.0)

    def test_synthesizes_wav_and_maps_emotion_before_request(self):
        events = {}
        client = mock.Mock()
        client.connected = True

        def register(event_name):
            return lambda callback: events.setdefault(event_name, callback)

        client.on.side_effect = register

        def emit(event_name, payload):
            self.assertEqual(event_name, "tts_request_gateway")
            self.assertIn("【happy2】", payload["text"])
            self.assertEqual(payload["language"], "English")
            self.assertEqual(payload["speed"], voice.AMIKA_TTS_SPEED)
            first_sentence = self._wav_bytes(b"\x01\x00" * 10)
            second_sentence = self._wav_bytes(b"\x02\x00" * 12)
            split = len(first_sentence) // 2
            for event in (
                {
                    "task_id": payload["task_id"],
                    "chunk_index": 0,
                    "chunk_count": 2,
                    "audio": first_sentence[:split],
                    "is_final": False,
                },
                {
                    "task_id": payload["task_id"],
                    "chunk_index": 1,
                    "chunk_count": 2,
                    "audio": first_sentence[split:],
                    "is_final": False,
                },
                {
                    "task_id": payload["task_id"],
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "audio": second_sentence,
                    "is_final": True,
                },
            ):
                events["audio_ready"](event)

        client.emit.side_effect = emit
        socketio = types.SimpleNamespace(Client=mock.Mock(return_value=client))
        adapter = types.SimpleNamespace(
            live_state=lambda: (True, "MekaHime_Amika", {"casual_happy": "happy2"}),
            remap_leading_tag=lambda text, _mapping: text.replace(
                "【casual_happy】", "【happy2】", 1
            ),
        )
        po_contract = types.SimpleNamespace(detect_lang=lambda _text: "en")
        import_module = mock.Mock(
            side_effect=lambda name: {
                "socketio": socketio,
                "openamika.speech.voice_adapter": adapter,
                "openamika.speech.po_contract": po_contract,
            }[name]
        )

        with (
            mock.patch.object(
                voice, "importlib", types.SimpleNamespace(import_module=import_module)
            ),
            mock.patch.object(voice, "AMIKA_TTS_URL", "http://127.0.0.1:8002"),
            mock.patch.object(client, "connect") as connect,
            mock.patch.object(client, "disconnect") as disconnect,
        ):
            with tempfile.TemporaryDirectory() as directory:
                wav_path = f"{directory}/amika-test.wav"
                voice._synthesize_with_amika("hello", "casual_happy", wav_path)
                with wave.open(wav_path, "rb") as wav_file:
                    self.assertEqual(wav_file.getnframes(), 22)
                    self.assertEqual(
                        wav_file.readframes(wav_file.getnframes()),
                        b"\x01\x00" * 10 + b"\x02\x00" * 12,
                    )

        connect.assert_called_once_with(
            "http://127.0.0.1:8002", transports=["websocket"], wait_timeout=10
        )
        disconnect.assert_called_once()

    def test_reports_openamika_server_error_message(self):
        events = {}
        client = mock.Mock()
        client.connected = True
        client.on.side_effect = lambda name: lambda callback: events.setdefault(name, callback)
        client.emit.side_effect = lambda _name, payload: events["error"]({
            "task_id": payload["task_id"],
            "error": "voice engine unavailable: out of memory",
        })
        socketio = types.SimpleNamespace(Client=mock.Mock(return_value=client))
        adapter = types.SimpleNamespace(live_state=lambda: (False, "MekaHime_Amika", {}))
        po_contract = types.SimpleNamespace(detect_lang=lambda _text: "en")
        import_module = mock.Mock(
            side_effect=lambda name: {
                "socketio": socketio,
                "openamika.speech.voice_adapter": adapter,
                "openamika.speech.po_contract": po_contract,
            }[name]
        )

        with (
            mock.patch.object(
                voice, "importlib", types.SimpleNamespace(import_module=import_module)
            ),
            mock.patch.object(client, "connect"),
            mock.patch.object(client, "disconnect") as disconnect,
        ):
            with self.assertRaises(RuntimeError) as raised:
                voice._synthesize_with_amika("hello", "neutral", "/tmp/amika-test.wav")

        self.assertEqual(
            str(raised.exception),
            "OpenAmika TTS failed: voice engine unavailable: out of memory",
        )
        disconnect.assert_called_once()

    def test_disconnect_during_synthesis_fails_immediately(self):
        events = {}
        client = mock.Mock()
        client.connected = True
        client.on.side_effect = lambda name: lambda callback: events.setdefault(name, callback)
        client.emit.side_effect = lambda _name, _payload: events["disconnect"]("transport close")
        socketio = types.SimpleNamespace(Client=mock.Mock(return_value=client))
        adapter = types.SimpleNamespace(live_state=lambda: (False, "MekaHime_Amika", {}))
        po_contract = types.SimpleNamespace(detect_lang=lambda _text: "en")
        import_module = mock.Mock(
            side_effect=lambda name: {
                "socketio": socketio,
                "openamika.speech.voice_adapter": adapter,
                "openamika.speech.po_contract": po_contract,
            }[name]
        )

        with (
            mock.patch.object(
                voice, "importlib", types.SimpleNamespace(import_module=import_module)
            ),
            mock.patch.object(client, "connect"),
            mock.patch.object(client, "disconnect") as disconnect,
        ):
            with self.assertRaisesRegex(RuntimeError, "connection closed before audio completed"):
                voice._synthesize_with_amika("hello", "neutral", "/tmp/amika-test.wav")

        disconnect.assert_called_once()

    def test_falls_back_to_piper_when_amika_is_unavailable(self):
        with (
            mock.patch.object(
                voice, "_synthesize_with_amika", side_effect=RuntimeError("offline")
            ),
            mock.patch.object(voice, "_synthesize_with_piper") as piper,
            mock.patch.object(voice, "_play_synthesized_audio") as play,
        ):
            voice._speak_text("hello", "happy", 0.5, "casual_happy")

        piper.assert_called_once()
        play.assert_called_once()

    def test_prefers_amika_without_calling_piper(self):
        with (
            mock.patch.object(voice, "_synthesize_with_amika") as amika,
            mock.patch.object(voice, "_synthesize_with_piper") as piper,
            mock.patch.object(voice, "_synthesize_with_cli") as piper_cli,
            mock.patch.object(voice, "_play_synthesized_audio") as play,
        ):
            voice._speak_text("hello", "happy", 0.5, "casual_happy")

        amika.assert_called_once()
        piper.assert_not_called()
        piper_cli.assert_not_called()
        play.assert_called_once()


if __name__ == "__main__":
    unittest.main()
