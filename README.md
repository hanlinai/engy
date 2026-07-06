# engy

**Verified inference for frontier open models.** Bittensor subnet 53.

engy serves frontier open models on consumer GPUs, with cryptographic
verification that the model you pay for is the model that ran. Each model's
weights and quantization are pinned by a published Merkle root. Miners sign a
commitment to the model's internal activations on every credited request, and
auditors challenge a random sample: the miner opens the commitment and the
opening is recomputed against the pinned weights. A failed opening is proof of
cheating, not bad luck. No TEEs, no trusted hardware. The proof pins the math,
not the machine.

- API and pricing: [engy.ai](https://engy.ai)
- Subnet: netuid 53 on Bittensor
- Company: [hanlin.ai](https://hanlin.ai)
- Contact: ning@engy.ai

Source (protocol, model specs and roots, miner client, incentive mechanism)
opens here as we approach open miner enrollment.
