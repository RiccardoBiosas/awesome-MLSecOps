# Site operations

## Local development

The site requires Node.js 22.12 or newer. CI uses Node.js 24.

```sh
nvm use
npm install
npm run dev
```

`npm run dev` and `npm run build` run `scripts/sync-readme.ts` first. The script parses the checked-out `README.md`, classifies eligible tool sections, removes known tracking parameters, validates the generated collection, and refuses to overwrite it if fewer than 60 entries are found.

To test a remote source explicitly:

```sh
README_SOURCE=https://raw.githubusercontent.com/RiccardoBiosas/awesome-MLSecOps/main/README.md npm run sync
```

## Deployment

`.github/workflows/deploy-pages.yml` builds and deploys with `withastro/action`. It runs on every push to `main`, on manual dispatch, and once daily so expired jobs and freshness metadata are rebuilt even when the repository has no commit.

The root `CNAME` remains the repository record. `public/CNAME` ensures Astro copies the same custom domain into the Pages artifact.

## Google Search Console

1. Create or open the `awesomemlsecops.com` domain property in Google Search Console.
2. Complete the DNS ownership challenge at the domain provider.
3. If Google also provides an HTML meta verification token, store only the token value in the repository Actions secret `GOOGLE_SITE_VERIFICATION`.
4. Deploy the site and verify that the token appears in the rendered `<head>`.
5. Submit `https://awesomemlsecops.com/sitemap.xml` in Search Console.

DNS verification and sitemap submission require access to the domain and Google account and cannot be completed by the build itself.

## Analytics

The static layout loads Plausible for `awesomemlsecops.com`. Create or connect that domain in Plausible before production measurement. The only local browser script is the event bridge required for these goals:

- `Newsletter Signup`
- `Repo Outbound`
- `Job Outbound`
- `Job Inquiry`
- `Sponsor Outbound`
- `Sponsor Page Visit`
- `Sponsor Inquiry`
- `Advisory Inquiry`

Plausible records referrers, which supports reporting on traffic from ChatGPT, Claude, Perplexity, Gemini, and other AI referral sources.

## Jobs

Jobs are maintained in `src/data/jobs.json`. Use the employer's canonical application URL and verified first-published date. Set an explicit `expires_date`; the shared `activeJobs()` utility removes expired records from rendered listings and `JobPosting` JSON-LD. The sitemap excludes `/jobs/` when no active listings remain.

Run the boundary tests after changing expiry logic:

```sh
npm test
```

## Sponsors

Sponsors are maintained only in `src/data/sponsors.json`. Keep at most three entries. An empty array collapses the row to a link to `/sponsor/`. Sponsor data never enters `src/data/tools.json`, and sponsor links render with `rel="sponsored"`.

## Newsletter

Archive entries are maintained in `src/data/newsletter.json` from canonical posts on The MLSecOps Hacker. Each record appears on `/newsletter/`, in `/rss.xml`, and in `/llms.txt`.