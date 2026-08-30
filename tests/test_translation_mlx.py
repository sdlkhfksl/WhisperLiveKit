"""Local MT lifecycle through the same worker used by WebSocket sessions."""

import asyncio
import sys
from types import SimpleNamespace

import pytest

from whisperlivekit.timed_objects import ASRToken, ChangeSpeaker, Silence, State
from whisperlivekit.translation_mlx_llm_mt import MlxLlmTranslation


@pytest.mark.asyncio
async def test_mlx_translation_drains_batches_boundaries_and_eof(monkeypatch):
    from whisperlivekit.audio_processor import SENTINEL, AudioProcessor

    calls = []

    def translate(self, text):
        calls.append((self._source_language, self._target_language, text))
        return text.upper()

    monkeypatch.setattr(MlxLlmTranslation, "_translate_text", translate)
    engine = MlxLlmTranslation(source_language="zh", target_language="en", warmup=False)
    from whisperlivekit.translation import session_translation_factory

    client = session_translation_factory(SimpleNamespace(lan="zh"), engine, "de", "fr")
    other = engine.new_session()
    processor = SimpleNamespace(
        translation_queue=asyncio.Queue(), translation=client,
        state=State(), lock=asyncio.Lock(),
    )
    for event in [
        ASRToken(0, 1, "One."), ASRToken(1, 2, " Two."), ASRToken(2, 3, " Three."),
        ASRToken(3, 4, "Pause"),
        Silence(start=4, end=5, is_starting=True, has_ended=True),
        ASRToken(5, 6, "Speaker"),
        ChangeSpeaker(start=6, speaker=2),
        ASRToken(6, 6.5, "Last"), ASRToken(6.5, 7, "words"),
        SENTINEL,
    ]:
        await processor.translation_queue.put(event)
    await asyncio.wait_for(AudioProcessor.translation_processor(processor), 5)
    assert " ".join(t.text for t in processor.state.new_translation) == "ONE. TWO. THREE. PAUSE SPEAKER LAST WORDS"
    assert not processor.state.new_translation_buffer.text
    assert client.finish()[0] is None  # no duplicate on a repeated drain
    assert all(source == "fr" and target == "de" for source, target, _ in calls)
    assert other.finish()[0] is None
    assert (other._source_language, other._target_language) == ("zh", "en")


def test_mlx_translation_keeps_failed_sentence_for_retry(monkeypatch):
    client = MlxLlmTranslation(warmup=False)
    failed = False

    def translate(text):
        nonlocal failed
        if text == "Two." and not failed:
            failed = True
            raise RuntimeError("decode failed")
        return text.upper()

    monkeypatch.setattr(client, "_translate_text", translate)
    client.insert_tokens([ASRToken(0, 1, "One."), ASRToken(1, 2, "Two."), ASRToken(2, 3, "Tail")])
    result, buffer = client.process()
    assert result.text == "ONE."
    assert "decode failed" in client.error
    assert not buffer.text
    result, buffer = client.finish()
    assert result.text == "TWO. TAIL"
    assert (result.start, result.end) == (1, 3)
    assert not client.error and not buffer.text
    assert client.finish()[0] is None


_HY_PLACEHOLDER = "<｜hy_place▁holder▁no▁2｜>"


class _Chunk:
    """Minimal stand-in for mlx_lm's GenerationResponse."""

    def __init__(self, token, text):
        self.token = token
        self.text = text


def _install_fake_mlx_lm(monkeypatch, chunks, consumed):
    """Inject stub ``mlx_lm`` / ``mlx_lm.sample_utils`` whose ``stream_generate``
    yields ``chunks`` and appends each yielded chunk to ``consumed`` (mutable),
    so the test can assert how much of the stream the decode loop consumed
    before breaking out."""
    import types

    def fake_stream_generate(model, tokenizer, prompt=None, max_tokens=None, **kw):
        for c in chunks:
            consumed.append(c)
            yield c

    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.stream_generate = fake_stream_generate
    sample_utils = types.ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = lambda **kw: None
    sample_utils.make_logits_processors = lambda **kw: []
    mlx_lm.sample_utils = sample_utils
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)


class _Tokenizer:
    """Stub tokenizer: configurable placeholder id sequence, no real vocab."""

    def __init__(self, placeholder_ids, eos_token=""):
        self._placeholder_ids = placeholder_ids
        self.eos_token = eos_token

    def encode(self, text, add_special_tokens=True):
        if text == _HY_PLACEHOLDER:
            return list(self._placeholder_ids)
        return [1]  # any non-placeholder prompt encodes to one id

    def decode(self, ids, skip_special_tokens=True):
        return ""

    def apply_chat_template(self, messages, add_generation_prompt=False):
        return [1, 2, 3]


def _backend_with_tokenizer(monkeypatch, tokenizer, chunks):
    """Backend whose model/tokenizer come from the stub registry; returns
    (backend, consumed) where consumed lists the chunks the decode loop
    actually pulled before stopping."""
    backend = MlxLlmTranslation(model_id="hy-mt2-1.8b-8bit", target_language="en", warmup=False)
    backend._eos_token = ""  # exercise only the placeholder-stop path
    consumed = []
    _install_fake_mlx_lm(monkeypatch, chunks, consumed)
    monkeypatch.setattr(
        type(backend), "_ensure_model", classmethod(lambda cls, config: (object(), tokenizer))
    )
    return backend, consumed


def test_early_stops_on_single_id_placeholder(monkeypatch):
    """Hy-MT2-1.8B case: placeholder is one token id — decode stops at that id
    and the hallucinated tail is never consumed."""
    tail_chunks = [_Chunk(9000 + i, "废" if i % 2 else "话") for i in range(20)]
    chunks = [
        _Chunk(500, "Hello"),
        _Chunk(501, " world"),
        _Chunk(120020, _HY_PLACEHOLDER),  # the placeholder, one id
        *tail_chunks,
    ]
    tok = _Tokenizer(placeholder_ids=[120020])
    backend, consumed = _backend_with_tokenizer(monkeypatch, tok, chunks)

    out = backend._translate_text("你好。")

    assert out == "Hello world"
    # Stop fired at the placeholder: only 3 chunks were consumed, not the tail.
    assert len(consumed) == 3


def test_early_stops_on_fragmented_placeholder(monkeypatch):
    """Fragmented (multi-id) placeholder: the rolling id window fires once the
    full sequence has been emitted."""
    frag_ids = [27, 15755, 250]
    chunks = [
        _Chunk(500, "Hi"),
        _Chunk(frag_ids[0], "<"),
        _Chunk(frag_ids[1], "｜hy"),
        _Chunk(frag_ids[2], "_place▁holder▁no▁2｜>"),
        _Chunk(7000, "junk"),
        _Chunk(9001, "更多"),
    ]
    tok = _Tokenizer(placeholder_ids=frag_ids)
    backend, consumed = _backend_with_tokenizer(monkeypatch, tok, chunks)

    out = backend._translate_text("你好。")

    assert out == "Hi"
    assert len(consumed) == 4  # stopped right after the last fragment


def test_no_stop_id_falls_back_to_full_decode_and_string_strip(monkeypatch):
    """Tokenizer that fragments the placeholder beyond the window cap: no
    in-loop stop (stream runs to completion), and the post-hoc string strip
    still truncates the hallucinated tail."""
    chunks = [
        _Chunk(500, "Hi "),
        _Chunk(700, "there "),
        _Chunk(901, _HY_PLACEHOLDER),
        _Chunk(902, "hallucinated tail"),
    ]
    tok = _Tokenizer(placeholder_ids=list(range(40)))  # >16 ids → no stop check
    backend, consumed = _backend_with_tokenizer(monkeypatch, tok, chunks)

    out = backend._translate_text("你好。")

    assert out == "Hi there"  # string strip still cut at the placeholder
    assert len(consumed) == len(chunks)  # nothing stopped in-loop (expected)


def test_early_stop_does_not_affect_clean_output(monkeypatch):
    """No placeholder in the stream: the loop runs to EOS-ish end and the
    output is untouched by the strip."""
    chunks = [_Chunk(500, "Hello "), _Chunk(501, "world")]
    tok = _Tokenizer(placeholder_ids=[120020])
    backend, consumed = _backend_with_tokenizer(monkeypatch, tok, chunks)

    out = backend._translate_text("你好。")

    assert out == "Hello world"
    assert len(consumed) == 2

