# Agent instructions for this repository

## `api-specifications/` is reference documentation

`api-specifications/{chat-completions,responses,messages}.md` document the LLM
API formats themselves. They are independent of this repository **and of each
other**, and each must read as if it were the only one published.

Rules for editing them:

- 🚫 **Never name this project, its scripts, or its capability codenames.** No
  `llmprobe`, no `probe_inference.py`, no `TCALL`/`STRM`/`GRAMT`/`TDEF`/`RJSON`/
  `REASN`/`CACH`. State the *behaviour*, not the tool that measured it.
- 🚫 **Never link to local files** — not `reports/`, not `CAPABILITIES.md`, not
  `~/bin/*`, not sibling prototypes, and not the other two specifications.
- 🧱 **One document, one protocol.** No cross-API comparison of any kind: no
  "unlike X", no "differs from X", no equivalence tables, no translation
  checklists, no links between the three documents. Every field — token
  accounting especially — is explained on its own terms, as a fact about the
  protocol being documented. Naming another protocol only to say what this one
  is not is still a comparison.
- 🔗 **Cite the internet inline.** Vendor API references, OpenAI/Anthropic
  guides, upstream source files on GitHub — as inline hyperlinks in the prose,
  where the claim is made.
- 🚫 **No `## Sources` section.** No trailing bibliography of any kind.
- 📐 **Keep the two-part structure**: Part 1 is the common, portable API; Part 2
  is provider quirks. A claim belongs in Part 1 only if it held on every
  endpoint tested.
- 🔬 **Observations over documentation.** Quirks are what the endpoints actually
  did. Quote the verbatim error string; where vendor docs and observed behaviour
  disagree, write down the observation.
- ✅ **Verified facts only — never what is claimed about them.** No "the most
  repeated claim", no "write-ups say", no "widely believed", no correcting a
  rumour by repeating it. State the measurement and the result. If something was
  not measured, leave it out rather than hedging it in.
- ✂️ **No effort narration.** How hard something was to find out is not a fact
  about the protocol: no "each of which costs an attempt", no "this took a while
  to pin down", no war stories. Describe what a section contains, not what it
  cost.
