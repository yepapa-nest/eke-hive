# Contributing

Thank you for looking! A few things that make contributions easy to accept:

- **Correctness first.** Every change to model math or kernels must keep the existing tests passing
  (`tools/test_cpu.sh`, and the GPU kernel tests that compare against the previous kernel). A new kernel needs a
  test that compares it against the kernel it replaces, bit for bit or with a stated tolerance and a reason.
- **Measure.** Performance changes need before/after numbers measured the same way (see
  [docs/performance.md](docs/performance.md#how-it-is-measured)), including the noise floor, and the machine they
  were measured on. New behaviour goes behind an environment switch until it is measured; once it is adopted,
  its value goes into `config/hive.env` (DeepSeek) or `config/glm.env` (GLM-5.3-Flash).
- **No new rejects on the request path.** Normalise unusual input at the entry point instead of failing the
  request.
- **English** for code, comments and documentation.
- **Sign-off.** Add `Signed-off-by:` to your commits (`git commit -s`) to certify the
  [Developer Certificate of Origin](https://developercertificate.org/). Contributions are accepted under the MIT
  License of this repository.

GPU validation runs on the maintainers' machine, so expect a delay between review and merge for changes that
need it. Please include the output of the tests you ran.
