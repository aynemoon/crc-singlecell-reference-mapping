# Training-log redaction record

Source: `logs/pr1-rerun.log` in the supplied `pr1_scientific_review.zip`.
The source log was not modified.

## Scope of redaction

The local home-directory username was replaced with the literal placeholder
`/home/REDACTED_USER` in **15 occurrences**. Only the `/home/<local-username>`
prefix changed. Repository-relative paths, package names, Python 3.12 directory,
source filenames, warning line numbers, and scVI/scANVI model paths are intact.
No other bytes were added, removed, normalized, or altered.

## Preservation checks

- The shareable file is exactly the original bytes with that one prefix substitution.
- Reversing the substitution locally reproduces the original file byte-for-byte.
- All numeric sequences are unchanged.
- Carriage returns, newlines, ANSI formatting, progress updates, epoch counts,
  losses, timing, warnings, informational messages and final score tables are preserved.
- Seed 0 messages, CPU/GPU availability messages, 200/40/100/100 epoch completion,
  both eval-mode warnings (42 and 85 modules), and `TRANSFER_EXIT=0` are intact.

## Sensitive-information inspection

The full log was inspected, including progress output and warnings. No credentials,
API keys, access tokens, email addresses, IP addresses, host identifiers, or
patient/donor/cell identifiers were identified. The only URL is a public PyPI
package link. The identified personal information was the local filesystem username.
This review cannot establish the absence of every conceivable secret format.

## Checksums (SHA-256, exact file bytes)

Original:
`e56a5b7d4d25a5a135580ba81533f008c02d68aaf64ba99d95aa26e0da824aa3`

Shareable copy:
`187c632dbd8927db6fcf58ce2f5a7f8d51af18552a1cd6a22465736c78cd1075`

This shareable copy has not been attached to GitHub or published in an Issue.
