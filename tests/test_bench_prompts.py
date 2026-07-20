from benchmark.prompts import synth_prompt


def test_prompt_length_is_close_to_requested_tokens():
    # Rough token accounting: one whitespace-separated word ~= one token.
    for want in (16, 128, 512):
        got = len(synth_prompt(seed=1, index=0, tokens=want).split())
        assert abs(got - want) <= want * 0.1


def test_prompts_differ_across_requests_to_defeat_prefix_cache():
    a = synth_prompt(seed=1, index=0, tokens=64)
    b = synth_prompt(seed=1, index=1, tokens=64)
    assert a != b
    # The divergence must be at the very front, or a shared prefix still caches.
    assert a.split()[0] != b.split()[0]


def test_same_seed_and_index_reproduces_the_same_prompt():
    assert synth_prompt(seed=7, index=3, tokens=32) == synth_prompt(seed=7, index=3, tokens=32)


def test_different_seeds_produce_different_prompts():
    assert synth_prompt(seed=1, index=0, tokens=32) != synth_prompt(seed=2, index=0, tokens=32)
