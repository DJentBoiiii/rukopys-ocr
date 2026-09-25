from model.reconstruction import ReconstructionModule, build_word_frequencies


def test_reconstruct_fixes_typo_within_edit_distance():
    freqs = build_word_frequencies(["привіт світ", "як справи", "привіт як"])
    module = ReconstructionModule(freqs, max_edit_distance=2)

    assert module.reconstruct("привт світ") == "привіт світ"


def test_reconstruct_leaves_unrelated_word_unchanged():
    freqs = build_word_frequencies(["привіт світ", "як справи"])
    module = ReconstructionModule(freqs, max_edit_distance=2)

    unrelated = "xqzwkjpl"
    assert module.reconstruct(unrelated) == unrelated
