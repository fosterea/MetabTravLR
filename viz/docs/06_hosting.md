# 06 — Hosting

How to put this app on the web. Foster's question: "can we set up a Firebase or
something, upload the files, and have it be easy and cheap?" Short answer: **yes.**

## The short answer

This app is a **static single-page app plus static JSON** — no backend, no database,
nothing server-side (see `01_architecture.md`, "Deployment posture"). `vite.config.ts`
already sets `base: './'`, so the build works from any path or subpath. That means **any
static host works**, setup is a few minutes, and at this scale it is **free or
near-free** on every option below. The whole payload is tiny: `public/data/` is under
**1 MB** today, and the compiled JS/CSS bundle is a few hundred KB — a rounding error
against every provider's free tier.

The build step is always the same:

```bash
cd viz
npm run build      # produces viz/dist/
```

`public/data/` is copied into `dist/` at build time (Vite's default for `public/`), so
**there is no separate data upload** — you deploy the one `dist/` folder and you're done.
Do not commit `dist/` (it's a build artifact).

## Firebase Hosting (the requested option)

Steps, run from `viz/`:

```bash
npm install -g firebase-tools     # or: npx firebase-tools ...
firebase login
firebase init hosting
#   - public directory:            dist
#   - configure as a single-page app (rewrite all URLs to /index.html):  Yes
#   - set up automatic builds/deploys with GitHub:                       optional
npm run build
firebase deploy
```

The two answers that matter: **public directory = `dist`**, and **yes to the SPA
rewrite** (so any path serves `index.html` — this app has no router today, but the
rewrite is harmless and future-proof). Deploy prints a `*.web.app` URL.

**Cost.** Firebase Hosting's free **Spark** plan includes a generous storage and
monthly-transfer allowance that a sub-1-MB site nowhere near approaches; you would only
start paying (on the pay-as-you-go **Blaze** plan) after a lot of storage or many GB of
monthly transfer — orders of magnitude beyond this app. Check Firebase's current pricing
page for the exact GB figures, but in practice this stays free.

## Alternatives worth knowing

All have generous free tiers that comfortably cover a static site this size.

- **GitHub Pages** — the repo is already on GitHub, so this is the least new tooling:
  enable Pages (or a `gh-pages` deploy action) pointing at the built `dist/`. `base: './'`
  already makes subpath hosting (`user.github.io/repo/`) work. **Tradeoff:** free Pages
  sites are **public only** (see privacy note below).
  - **✅ Now wired up (2026-07-17).** `.github/workflows/deploy-viz.yml` builds `viz/` and
    deploys `viz/dist` to Pages on every push to `release` touching `viz/**` (and on manual
    "Run workflow"). **One-time step for Foster:** repo **Settings → Pages → Source: GitHub
    Actions**. Then the site publishes at `https://fosterea.github.io/MetabTravLR/`. Because
    the data is bundled into the build, there is no separate data upload.
- **Cloudflare Pages** — connect the GitHub repo, set build command `npm run build` and
  output dir `dist`. Fast global CDN, very generous free tier. **Tradeoff:** a separate
  Cloudflare account/dashboard to manage.
- **Netlify** — same shape: connect repo, build `npm run build`, publish `dist`. Very
  smooth UX, deploy previews on PRs. **Tradeoff:** another account; free-tier bandwidth
  caps exist but are far above this app's needs.

**Recommendation.** If the data can be public, use **GitHub Pages** — the repo is already
there, it's the fewest moving parts, and `base: './'` already works. If the data should
**not** be world-readable, use **Firebase Hosting** (or Cloudflare/Netlify) so you can put
it behind access control.

## Access control / privacy

This is research data, so decide whether it should be world-readable before choosing.

- **GitHub Pages** free tier is **public only** — anyone with the URL can read the data.
- **Firebase Hosting**, **Cloudflare Pages**, and **Netlify** can gate a site behind
  authentication or a password, though generally on paid or higher tiers / with extra
  setup (e.g. Cloudflare Access, Netlify password protection, or a Firebase auth check).

If the data is fine to share openly, this is a non-issue and Pages is simplest. If not,
pick one of the gated hosts and turn on access control. Nothing here encrypts the JSON —
gating is at the host/request layer.

## When the data grows (future)

Today the data is small and **bundled into the build**, which is the simplest possible
setup. If datasets later get large (many datasets, or big edge bundles), the option is to
host `public/data/` **separately** — object storage or a CDN (e.g. a storage bucket,
Cloudflare R2, S3) — and have the app `fetch()` it at runtime instead of baking it into
`dist/`. That keeps the build small and lets data update without a rebuild. This is a
**future** consideration only; nothing about the current architecture needs it now.

## Cost bottom line

At this scale the app is **effectively free to host anywhere**. It's under a megabyte of
data plus a small JS bundle, served as static files — every option here (Firebase, GitHub
Pages, Cloudflare Pages, Netlify) has a free tier that this site won't come close to
exhausting. You would only ever pay if traffic or data grew by orders of magnitude, and
even then it would be cents. Pick the host that matches your **privacy** need (public →
GitHub Pages; gated → Firebase/Cloudflare/Netlify), not the price.
