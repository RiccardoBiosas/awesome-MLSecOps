# Contributing to Awesome MLSecOps

Thank you for contributing to **Awesome MLSecOps**, a curated collection of tools, research, standards, threat models, and technical resources for securing machine-learning and AI systems.

**MLSecOps (Machine Learning Security Operations)** integrates security engineering, threat modeling, testing, supply-chain controls, monitoring, and incident response across the machine-learning lifecycle. It protects data, models, pipelines, infrastructure, LLM applications, and AI agents against adversarial attacks, compromise, privacy leakage, model theft, and abuse.

This is an editorially curated technical resource - not a comprehensive vendor directory or promotional channel. Inclusion depends on relevance, evidence, maturity, and practical value.

## Technical Scope

Eligible resources should address at least one concrete MLSecOps or AI-security domain:

- **Adversarial machine learning:** evasion, poisoning, backdoors, model extraction, inversion, membership inference, privacy leakage, and robustness.
- **LLM and generative-AI security:** prompt injection, jailbreaks, sensitive-data exposure, insecure output handling, RAG poisoning, guardrails, and red teaming.
- **AI agent security:** identity, authorization, tool security, memory poisoning, excessive agency, sandboxing, MCP security, and runtime policy enforcement.
- **AI supply-chain security:** model scanning, unsafe serialization, artifact signing, provenance, ML bills of materials, dataset integrity, and dependency risk.
- **MLOps infrastructure security:** notebooks, training pipelines, model registries, feature stores, secrets, cloud and Kubernetes workloads, and inference endpoints.
- **AI threat modeling and assurance:** security frameworks, benchmarks, testing methodologies, incident analysis, architecture, control mapping, and technical governance.

Resources focused primarily on general AI, fairness, explainability, or governance must demonstrate a direct security application.

## Accepted Resources

The list may include:

- Open-source security tools
- Security-focused commercial tools
- Proof-of-concept attacks and defenses
- Peer-reviewed papers and substantive technical reports
- Standards, frameworks, benchmarks, and datasets
- Threat models, security advisories, and vulnerability research
- Technical tutorials, books, and practitioner communities

A resource is not eligible merely because it uses AI, machine learning, or the term “AI security.”

## Inclusion Criteria

A submission must:

1. Address a specific AI or machine-learning security problem.
2. Be publicly usable or independently evaluable.
3. Provide meaningful technical documentation or methodology.
4. Link to the canonical project, product, or publication.
5. Have a factual, one-sentence description.
6. Demonstrate reasonable maintenance or lasting research value.
7. Disclose the contributor’s affiliation.
8. Not duplicate an existing entry.

Useful supporting evidence includes source code, releases, tests, architecture documentation, reproducible demonstrations, public trials, APIs, benchmarks, research artifacts, independent integrations, or technical evaluations.

Landing pages containing only marketing copy, waitlists, or contact forms are insufficient.

## Project Maturity

Projects less than **90 days old** are normally deferred until there is enough evidence to evaluate their utility and maintenance.

An early-stage project may qualify when it provides strong evidence such as:

- A working, documented release
- A clear MLSecOps use case
- Reproducible technical results
- Independent contributors or users
- An associated paper, audit, or credible evaluation
- Integration with an established security or MLOps ecosystem

GitHub stars are one adoption signal, not a quality threshold. A project with fewer than 30 stars may qualify, while a popular project may still be declined.

## Open-Source Projects

Open-source submissions should normally provide:

- A recognized license
- Installation and usage documentation
- A working release or reproducible commit
- Examples, tests, or documented behavior
- An issue tracker or support channel
- A security-reporting process when handling untrusted or sensitive input

Source availability alone does not establish technical quality or relevance.

## Commercial Tools

Commercial products are eligible only for the **Commercial Tools** section and must:

- Address a substantial MLSecOps or AI-security use case
- Represent a working product rather than a concept or waitlist
- Provide public technical documentation
- Identify supported systems, integrations, or deployment models
- Offer a practical evaluation path, such as a demo, trial, API, or detailed methodology
- Be described using verifiable capabilities rather than vendor claims

Generic AI, cloud, cybersecurity, or MLOps platforms do not qualify solely because they include a limited AI-security feature.

Free and paid products are evaluated using the same editorial standards.

## Self-Promotion and Disclosure

Self-submissions are allowed, but all material affiliations must be disclosed. This includes being a founder, employee, maintainer, contributor, contractor, advisor, investor, sponsor, or representative.

Add one of these statements to the pull request:

> **Disclosure:** I am affiliated with this project as [relationship].

> **Disclosure:** I have no material affiliation with this project.

Undisclosed self-promotion may result in rejection or reassessment of an existing entry. Affiliation does not automatically disqualify a resource, but it requires neutral wording and independently verifiable evidence.

Submit only one product per pull request.

## Sponsorship and Editorial Independence

Sponsorship supports repository maintenance but does not guarantee:

- Inclusion
- Preferential placement
- Favorable wording
- Editorial endorsement
- Permanent listing

Sponsored resources must independently satisfy the same technical criteria as all other submissions. Material commercial relationships must be disclosed.

Awesome MLSecOps does not sell undisclosed placement within editorial categories.

## Description Style

Descriptions must be factual, neutral, and limited to one sentence.

### Good

```markdown
| [Model Scanner](https://example.com/) | Open-source scanner for detecting unsafe serialization constructs in machine-learning model artifacts. |
```

```markdown
- [RAG Poisoning Study](https://example.com/) - Research evaluating retrieval-poisoning attacks against document-grounded LLM applications.
```

### Not acceptable

```markdown
| [Model Scanner](https://example.com/) | The world's leading enterprise-grade AI-security platform. |
```

Avoid unsupported language such as:

- Best
- Leading
- Revolutionary
- Industry-first
- Most comprehensive
- State-of-the-art
- Enterprise-grade
- Complete protection

Do not copy vendor taglines or include calls to action.

## Link Requirements

Links must:

- Use the canonical HTTPS destination
- Exclude referral, affiliate, and campaign parameters
- Avoid URL shorteners
- Resolve without unnecessary redirects
- Point to stable documentation or publication pages where possible

Do not delete an existing entry solely because its URL changed. Find and propose the canonical replacement when one exists.

A single `403`, `429`, timeout, or server error does not prove that a resource is permanently unavailable.

## Pull Request Requirements

Each pull request must:

- Add one resource or address one focused concern
- Explain the concrete MLSecOps security use case
- Include technical and maturity evidence
- Disclose every material affiliation
- Use the existing table or list format
- Preserve heading anchors and HTML anchor aliases
- Avoid unrelated formatting or structural changes
- Pass automated link and formatting checks

Include this information in the PR description:

| Field | Required information |
|---|---|
| Security use case | Security task performed |
| Protected component | Data, model, pipeline, agent, application, or infrastructure |
| Threat coverage | Attacks, weaknesses, or control objectives addressed |
| Evaluation | How practitioners can independently assess the resource |
| Maturity | Launch date, releases, maintenance, and adoption evidence |
| Affiliation | Contributor’s relationship to the resource |

Recommended PR title:

```text
Add <resource> to <section>
```

## Exclusion Criteria

Submissions will normally be declined when they are:

- Outside the technical scope of MLSecOps
- Primarily promotional or impossible to evaluate independently
- Unreleased concepts, waitlists, or placeholder repositories
- Generic products with incidental AI-security functionality
- Duplicates, link farms, or automatically generated collections
- Undocumented, abandoned, deceptive, or misleading
- Submitted through tracking or referral URLs
- Described using unverifiable security claims

Offensive security resources may qualify when they clearly support authorized research, testing, education, or defensive evaluation.

## Review Outcomes

The maintainer may:

- **Accept** a submission that meets the technical and editorial criteria.
- **Request changes** to its evidence, description, disclosure, or placement.
- **Defer** a relevant project until it develops sufficient maturity.
- **Decline** an out-of-scope, promotional, duplicative, or insufficiently technical resource.

Meeting every documented criterion does not guarantee inclusion. Final decisions remain subject to editorial judgment.

## Removal and Reassessment

An existing resource may be corrected, moved, or removed when it:

- Is permanently unavailable without a canonical replacement
- Is abandoned and no longer practically useful
- Changes scope and is no longer relevant
- Becomes misleading, compromised, or primarily promotional
- Makes repeated unsupported security claims
- Was included using inaccurate or undisclosed information

Temporary outages, inactivity alone, low star counts, or one failed automated check are insufficient grounds for removal.

## Contributor Checklist

Before requesting review, confirm that:

- [ ] The resource addresses a concrete MLSecOps or AI-security problem.
- [ ] It is publicly usable or independently evaluable.
- [ ] Technical and maturity evidence is included.
- [ ] The canonical URL contains no tracking parameters.
- [ ] The description is factual and limited to one sentence.
- [ ] The resource is not already listed.
- [ ] It is placed in the correct section and format.
- [ ] All affiliations are disclosed.
- [ ] Existing headings and anchors remain unchanged.
- [ ] The pull request contains one focused concern.

If you are unsure whether a resource qualifies, open an issue with its URL, security use case, maturity evidence, intended section, and affiliation disclosure.

Thank you for helping keep Awesome MLSecOps technically rigorous, independent, current, and useful to the AI-security community.
