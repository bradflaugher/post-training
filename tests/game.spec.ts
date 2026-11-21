import { test, expect, type Page } from '@playwright/test';

// Tiles carry their true lobe in data-lobe-id, so specs can clamp exact groups without spoiling the UI
async function readLobes(page: Page): Promise<Record<string, string[]>> {
  await expect(page.locator('.token-tile')).toHaveCount(16);
  return page.evaluate(() => {
    const out: Record<string, string[]> = {};
    document.querySelectorAll<HTMLElement>('.token-tile').forEach(el => {
      const id = el.dataset.lobeId || '';
      (out[id] ||= []).push(el.dataset.word || '');
    });
    return out;
  });
}

async function clamp(page: Page, words: string[]): Promise<void> {
  for (const w of words) {
    await page.locator(`.token-tile[data-word="${w}"]`).click();
  }
}

test.describe('POST-TRAINING — Neural Vivisection Word Game', () => {
  test('renders 16 mystery tokens with zero category spoilers, keeps header on one line, and fits mobile & laptop viewports', async ({ page }) => {
    // Check 1280x800 laptop header is on one horizontal line in Ward mode
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto('/index.html?lite=1&seed=1');
    await expect(page.locator('#mode-btn')).toHaveAttribute('data-mode', 'ward');
    await expect(page.locator('#case-title')).toContainText('CASE #001');

    const laptopHeaderAligned = await page.evaluate(() => {
      const left = document.querySelector('.brand-left')?.getBoundingClientRect();
      const right = document.querySelector('.header-controls')?.getBoundingClientRect();
      if (!left || !right) return false;
      return Math.abs(left.top - right.top) < 16 && right.left > left.right;
    });
    expect(laptopHeaderAligned).toBe(true);

    // Check 375x667 mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    const mobileHeaderAligned = await page.evaluate(() => {
      const left = document.querySelector('.brand-left')?.getBoundingClientRect();
      const right = document.querySelector('.header-controls')?.getBoundingClientRect();
      if (!left || !right) return false;
      return Math.abs(left.top - right.top) < 16 && right.left >= left.right - 2;
    });
    expect(mobileHeaderAligned).toBe(true);

    // No category or probe spoilers are shown before a category is solved
    await expect(page.locator('.solved-lobe')).toHaveCount(0);
    await expect(page.locator('#active-probe-title')).toContainText('0/4 LOBES');

    const tiles = page.locator('.token-tile');
    await expect(tiles).toHaveCount(16);

    // Reimagined token cards show Cortical Token ID + EKG waveform + UNCLAMPED badge
    const firstStatus = await tiles.first().locator('.token-clamp-badge').textContent();
    expect(firstStatus).toContain('UNCLAMPED');
    await expect(tiles.first().locator('svg.token-ekg')).toBeVisible();

    // Verify the entire game fits on a small phone screen (375x667) without vertical scrolling or clipping
    const fitsViewport = await page.evaluate(() => {
      return document.documentElement.scrollHeight <= window.innerHeight;
    });
    expect(fitsViewport).toBe(true);
    const footerBox = await page.locator('.footer-nav').boundingBox();
    expect(footerBox).not.toBeNull();
    expect((footerBox?.y ?? 0) + (footerBox?.height ?? 0)).toBeLessThanOrEqual(667);

    // Verify bottom scrolling ticker has no LED badge or light
    await expect(page.locator('.ticker-badge')).toHaveCount(0);
    await expect(page.locator('.telemetry-bar')).not.toContainText('● LED');
  });

  test('can select 4 matching tokens into the Synaptic Clamp Bus, amputate a cortical lobe, and reveal its probe & SLM probabilities', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/index.html?lite=1&seed=1');

    // Clamp all 4 tokens of the ice_cream lobe
    const lobes = await readLobes(page);
    await clamp(page, lobes.ice_cream);

    const selectedTiles = page.locator('.token-tile.selected');
    await expect(selectedTiles).toHaveCount(4);
    await expect(page.locator('.clamp-slot.filled')).toHaveCount(4);

    const submitBtn = page.locator('#submit-btn');
    await expect(submitBtn).toBeEnabled();
    await submitBtn.click();

    // 1 lobe solved, 12 tokens left on grid, and the solved lobe reveals its probe + probabilities + logits
    const solvedLobe = page.locator('.solved-lobe');
    await expect(solvedLobe).toHaveCount(1);
    await expect(solvedLobe.first()).toContainText('the best ice cream flavor is [MASK].');
    await expect(solvedLobe.first()).toContainText('%');
    // The declassified banner is stamped with its probe letter and entanglement tier (I–IV)
    await expect(solvedLobe.first().locator('.tier-tag')).toHaveText(/^[αβγδ]·(I|II|III|IV)$/);
    await expect(solvedLobe.first()).toHaveAttribute('data-tier', /^[1-4]$/);
    await expect(page.locator('#active-probe-title')).toContainText('1/4 LOBES');
    await expect(page.locator('.token-tile')).toHaveCount(12);

    // Still fits phone screen without scrolling after solving a category
    const fitsViewport = await page.evaluate(() => {
      return document.documentElement.scrollHeight <= window.innerHeight;
    });
    expect(fitsViewport).toBe(true);
  });

  test('detects One-Away (3/4 synapses aligned), decrements defibrillator charge on mistake, and blocks duplicate guesses', async ({ page }) => {
    await page.goto('/index.html?lite=1&seed=1');

    // Clamp 3 correct tokens of one lobe
    const lobes = await readLobes(page);
    await clamp(page, lobes.ice_cream.slice(0, 3));

    // Pick 1 token from a different lobe
    const nonMatchingTile = page.locator('.token-tile:not([data-lobe-id="ice_cream"])').first();
    await nonMatchingTile.click();

    const submitBtn = page.locator('#submit-btn');
    await submitBtn.click();
    await expect(page.locator('#telemetry-msg')).toContainText('3/4 SYNAPSES ALIGNED');
    await expect(page.locator('.volt-pip.lost')).toHaveCount(1);

    // Same 4 tokens remain selected, but #submit-btn is now disabled with "ALREADY TRIED"
    await expect(submitBtn).toBeDisabled();
    await expect(submitBtn).toHaveText('ALREADY TRIED');

    // Mashing Enter or re-selecting the same 4 tokens must not deduct another defibrillator charge
    await page.keyboard.press('Enter');
    await expect(page.locator('.volt-pip.lost')).toHaveCount(1);
  });

  test('completes a full surgical case and shows the Post-Op Autopsy Report modal without colored emoji squares or button overflow on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/index.html?lite=1&seed=1');

    const submitBtn = page.locator('#submit-btn');
    const lobes = await readLobes(page);
    const lobeIds = Object.keys(lobes);

    // One deliberate misfire first, so the report has a mixed row to draw
    await clamp(page, [...lobes[lobeIds[0]].slice(0, 3), lobes[lobeIds[1]][0]]);
    await submitBtn.click();
    await page.locator('#clear-btn').click();

    for (const id of lobeIds) {
      await clamp(page, lobes[id]);
      await submitBtn.click();
    }

    await expect(page.locator('.solved-lobe')).toHaveCount(4);
    await expect(page.locator('#debrief-dialog')).toBeVisible();
    await expect(page.locator('#debrief-summary')).toContainText('SURGERY SUCCESSFUL');
    await expect(page.locator('#debrief-lobes-list .debrief-lobe-item')).toHaveCount(4);
    const shareText = (await page.locator('#debrief-grid-text').textContent()) ?? '';
    expect(shareText).toContain('mode=ward');
    expect(shareText).not.toMatch(/[🟥🟩🟨🟪]/u);
    // Spoiler-free guess grid: 5 attempts, each drawn in entanglement-tier glyphs
    expect(shareText.split('\n').filter(line => /^[░▒▓█]{4}$/u.test(line))).toHaveLength(5);

    // The autopsy teaches: XP breakdown, a 16×4 logit heatmap, a declassified field note, and the surgeon's record
    // EASY base 100 + 3 defib charges left × 50 (+50 if the tier-IV lobe happened to fall first)
    await expect(page.locator('#debrief-score')).toContainText(/\+(250|300) XP/);
    await expect(page.locator('#debrief-heatmap .hm-cell')).toHaveCount(64);
    await expect(page.locator('#debrief-heatmap .hm-cell.own')).toHaveCount(16);
    await expect(page.locator('#debrief-field-note')).toContainText('FIELD NOTE 01/20 — DECLASSIFIED: TOKENS');
    await expect(page.locator('#debrief-stats-grid')).toContainText('1/20');
    await expect(page.locator('#streak-badge')).toHaveText('STREAK: 1');

    // The share report wraps inside its box on a phone: nothing in the autopsy body runs off-screen sideways,
    // and every guess-grid glyph sits in an equal-width cell
    const reportFits = await page.evaluate(() => {
      const pre = document.querySelector('#debrief-grid-text') as HTMLElement;
      const body = document.querySelector('#debrief-dialog .dlg-body') as HTMLElement;
      const widths = Array.from(pre.querySelectorAll('.tier-glyph'), el => Math.round(el.getBoundingClientRect().width * 10));
      return (
        pre.scrollWidth <= pre.clientWidth + 1 &&
        body.scrollWidth <= body.clientWidth + 1 &&
        widths.length >= 20 &&
        new Set(widths).size === 1
      );
    });
    expect(reportFits).toBe(true);

    // Verify #copy-report-btn and #next-case-btn stay inside the dialog and do not overflow their boxes on mobile
    const buttonsFit = await page.evaluate(() => {
      const dlg = document.querySelector('#debrief-dialog')?.getBoundingClientRect();
      const copyBtn = document.querySelector('#copy-report-btn') as HTMLElement | null;
      const nextBtn = document.querySelector('#next-case-btn') as HTMLElement | null;
      if (!dlg || !copyBtn || !nextBtn) return false;
      const copyRect = copyBtn.getBoundingClientRect();
      const nextRect = nextBtn.getBoundingClientRect();
      return (
        copyRect.width >= 100 &&
        copyBtn.scrollWidth <= copyBtn.clientWidth + 2 &&
        copyRect.left >= dlg.left &&
        copyRect.right <= dlg.right &&
        nextRect.left >= dlg.left &&
        nextRect.right <= dlg.right
      );
    });
    expect(buttonsFit).toBe(true);

    // Closing the report must not strand the player: the surgical tools give way to a pulsing NEXT PATIENT
    // and a way back into the report, and the whole console still fits the phone
    await page.locator('#debrief-close-btn').click();
    await expect(page.locator('#debrief-dialog')).toBeHidden();
    for (const id of ['#shuffle-btn', '#biopsy-btn', '#clear-btn', '#submit-btn']) {
      await expect(page.locator(id)).toBeHidden();
    }
    await expect(page.locator('#next-patient-btn')).toBeVisible();
    await expect(page.locator('#report-btn')).toBeVisible();
    await expect(page.locator('#telemetry-msg')).toContainText('NEXT PATIENT');
    const closedFits = await page.evaluate(() =>
      document.documentElement.scrollWidth <= window.innerWidth && document.documentElement.scrollHeight <= window.innerHeight
    );
    expect(closedFits).toBe(true);

    // Keyboard: Enter on the focused AUTOPSY REPORT button opens the report rather than skipping to a new board
    await page.locator('#report-btn').focus();
    await page.keyboard.press('Enter');
    await expect(page.locator('#debrief-dialog')).toBeVisible();
    await expect(page.locator('#debrief-summary')).toContainText('SURGERY SUCCESSFUL');
    await expect(page.locator('#seed-btn')).toContainText('#0001');
    await page.locator('#debrief-close-btn').click();

    // NEXT PATIENT deals a fresh board and hands the tools back
    await page.locator('#next-patient-btn').click();
    await expect(page.locator('.token-tile')).toHaveCount(16);
    await expect(page.locator('.solved-lobe')).toHaveCount(0);
    await expect(page.locator('#next-patient-btn')).toBeHidden();
    await expect(page.locator('#shuffle-btn')).toBeVisible();
    await expect(page.locator('#submit-btn')).toBeVisible();
    await expect(page.locator('#seed-btn')).not.toContainText('#0001');
    await expect(page.locator('#telemetry-msg')).not.toContainText('CASE CLOSED');
  });

  test('advancing during the closing animation never reopens the report over the fresh board', async ({ page }) => {
    await page.goto('/index.html?lite=1&seed=1');
    const lobes = await readLobes(page);
    for (const id of Object.keys(lobes)) {
      await clamp(page, lobes[id]);
      await page.locator('#submit-btn').click();
    }
    // NEXT PATIENT is already live while the report's opening delay is still pending
    await page.locator('#next-patient-btn').click();
    await expect(page.locator('.token-tile')).toHaveCount(16);
    await page.waitForTimeout(900);
    await expect(page.locator('#debrief-dialog')).toBeHidden();
    await expect(page.locator('.solved-lobe')).toHaveCount(0);
  });

  test('cycles cosine-distance difficulty grades, rolls random seeds on refresh/grade change, and synthesizes billions of procedural seeds', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/index.html?lite=1&seed=3923223191');
    await expect(page.locator('#seed-btn')).toContainText('SEED: #3923223191');
    await expect(page.locator('#grade-btn')).toContainText('MED · Δ');

    // Verify #grade-btn is completely visible and unclipped on a 375px screen even with a 10-digit seed
    const gradeBox = await page.locator('#grade-btn').boundingBox();
    expect(gradeBox).not.toBeNull();
    expect((gradeBox?.x ?? 0) + (gradeBox?.width ?? 0)).toBeLessThanOrEqual(375);

    const wordsA = await page.locator('.token-word').allTextContents();

    await page.locator('#grade-btn').click();
    await expect(page.locator('#grade-btn')).toContainText('HARD · Δ');
    await expect(page.locator('.token-tile')).toHaveCount(16);
    const hardBadgeText = (await page.locator('#grade-btn').textContent()) ?? '';
    const hardDelta = parseFloat(hardBadgeText.split('Δ')[1] ?? '1');
    expect(hardDelta).toBeLessThan(0.62);

    await page.locator('#grade-btn').click();
    await expect(page.locator('#grade-btn')).toContainText('EASY · Δ');
    await expect(page.locator('.token-tile')).toHaveCount(16);
    const easyBadgeText = (await page.locator('#grade-btn').textContent()) ?? '';
    const easyDelta = parseFloat(easyBadgeText.split('Δ')[1] ?? '0');
    expect(easyDelta).toBeGreaterThanOrEqual(0.8);

    // Verify clicking #new-case-btn across 10 consecutive games (pushing recentLobeIds past its 32-entry cap)
    // never repeats a category from any of the preceding 7 games (every 8-game window has 32 distinct lobes)
    const seedBeforeRefresh = await page.locator('#seed-btn').textContent();
    const firstBoardLobes = await page.evaluate(() =>
      Array.from(new Set(Array.from(document.querySelectorAll('.token-tile')).map(el => (el as HTMLElement).dataset.lobeId || '')))
    );
    expect(firstBoardLobes.length).toBe(4);
    const boardHistory: string[][] = [firstBoardLobes];
    for (let g = 0; g < 10; g++) {
      await page.locator('#new-case-btn').click();
      await expect(page.locator('#grade-btn')).toContainText('EASY · Δ');
      const curLobes = await page.evaluate(() =>
        Array.from(new Set(Array.from(document.querySelectorAll('.token-tile')).map(el => (el as HTMLElement).dataset.lobeId || '')))
      );
      expect(curLobes.length).toBe(4);
      const recentWindowLobes = new Set(boardHistory.slice(-7).flat());
      for (const lid of curLobes) {
        expect(recentWindowLobes.has(lid)).toBe(false);
      }
      boardHistory.push(curLobes);
    }
    const seedAfterRefresh = await page.locator('#seed-btn').textContent();
    expect(seedAfterRefresh).not.toBe(seedBeforeRefresh);

    // Verify 32-bit Math.imul seed hashing distinguishes large seeds (1588089239 vs 3923223191)
    await page.goto('/index.html?lite=1&seed=1588089239');
    await expect(page.locator('.token-tile')).toHaveCount(16);
    const wordsB = await page.locator('.token-word').allTextContents();
    expect(wordsA.join(',')).not.toBe(wordsB.join(','));

    // Verify shared seed=1&grade=clean link preserves canonical seed 1 board
    await page.goto('/index.html?lite=1&seed=1&grade=clean');
    await expect(page.locator('.token-tile')).toHaveCount(16);
    const canonicalWords = await page.locator('.token-word').allTextContents();
    expect(canonicalWords).toContain('vanilla');
  });

  test('biopsies a clamped token into a 4-probe activation signature, limited to 3 needles per case', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/index.html?lite=1&seed=1');
    const biopsyBtn = page.locator('#biopsy-btn');
    await expect(page.locator('.token-tile')).toHaveCount(16);
    await expect(biopsyBtn).toHaveText(/BIOPSY 3$/);

    // Nothing clamped: no needle is spent
    await biopsyBtn.click();
    await expect(biopsyBtn).toHaveText(/BIOPSY 3$/);
    await expect(page.locator('#telemetry-msg')).toContainText('CLAMP A TOKEN FIRST');

    const lobes = await readLobes(page);
    const [first, second, third, fourth] = lobes.ice_cream;
    await clamp(page, [first]);
    await biopsyBtn.click();
    await expect(biopsyBtn).toHaveText(/BIOPSY 2$/);
    const tile = page.locator(`.token-tile[data-word="${first}"]`);
    await expect(tile).toHaveClass(/biopsied/);
    await expect(tile.locator('.token-sig i')).toHaveCount(4);
    await expect(page.locator('#telemetry-msg')).toContainText(`BIOPSY "${first.toUpperCase()}"`);

    // The signature is a probability distribution over the 4 hidden probes, and two tokens of one lobe peak on the same probe
    const peakOf = async (word: string) => {
      const title = (await page.locator(`.token-tile[data-word="${word}"]`).getAttribute('title')) ?? '';
      const pcts = Array.from(title.matchAll(/(\d+)%/g), m => Number(m[1]));
      expect(pcts).toHaveLength(4);
      expect(Math.abs(pcts.reduce((a, b) => a + b, 0) - 100)).toBeLessThanOrEqual(2);
      return pcts.indexOf(Math.max(...pcts));
    };
    await clamp(page, [second]);
    await biopsyBtn.click();
    expect(await peakOf(second)).toBe(await peakOf(first));

    // Re-biopsying the same token is refused; the third needle empties the tray
    await page.locator('#clear-btn').click();
    await clamp(page, [first]);
    await biopsyBtn.click();
    await expect(biopsyBtn).toHaveText(/BIOPSY 1$/);
    await clamp(page, [third]);
    await biopsyBtn.click();
    await expect(biopsyBtn).toHaveText(/BIOPSY 0$/);
    await expect(biopsyBtn).toBeDisabled();
    await clamp(page, [second, fourth]);
    await page.locator('#submit-btn').click();
    await expect(page.locator('.solved-lobe')).toHaveCount(1);

    // The whole console still fits a phone with no sideways overflow
    const noOverflow = await page.evaluate(() =>
      document.documentElement.scrollWidth <= window.innerWidth && document.documentElement.scrollHeight <= window.innerHeight
    );
    expect(noOverflow).toBe(true);
  });

  test('deals a deterministic Daily Case, files it once, and wheels returning surgeons to today\'s case', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    // 2024-01-01 was a Monday: the week opens with an EASY board
    await page.goto('/index.html?daily=2024-01-01');
    await expect(page.locator('#case-title')).toContainText('DAILY 2024-01-01');
    await expect(page.locator('#seed-btn')).toContainText('2024-01-01');
    await expect(page.locator('#grade-btn')).toContainText('EASY');
    await expect(page.locator('.token-tile')).toHaveCount(16);
    const lobes = await readLobes(page);
    const boardKey = (m: Record<string, string[]>) => Object.keys(m).sort().map(k => `${k}:${[...m[k]].sort().join(',')}`).join('|');

    // One shot means one shot: a spent charge, a spent needle and a severed lobe all survive a refresh
    const ids = Object.keys(lobes);
    await clamp(page, [...lobes[ids[0]].slice(0, 3), lobes[ids[1]][0]]);
    await page.locator('#submit-btn').click();
    await expect(page.locator('.volt-pip.lost')).toHaveCount(1);
    await page.locator('#biopsy-btn').click();
    await expect(page.locator('#biopsy-btn')).toHaveText(/BIOPSY 2$/);
    await page.locator('#clear-btn').click();
    await clamp(page, lobes[ids[0]]);
    await page.locator('#submit-btn').click();
    await expect(page.locator('.solved-lobe')).toHaveCount(1);
    // A detour through another unfiled date must not wipe this attempt either
    await page.goto('/index.html?daily=2024-01-02');
    const detour = await readLobes(page);
    const detourIds = Object.keys(detour);
    await clamp(page, [...detour[detourIds[0]].slice(0, 3), detour[detourIds[1]][0]]);
    await page.locator('#submit-btn').click();
    await expect(page.locator('.volt-pip.lost')).toHaveCount(1);
    await page.goto('/index.html?daily=2024-01-01');
    await expect(page.locator('.token-tile')).toHaveCount(12);
    await expect(page.locator('.solved-lobe')).toHaveCount(1);
    await expect(page.locator('.volt-pip.lost')).toHaveCount(1);
    await expect(page.locator('#biopsy-btn')).toHaveText(/BIOPSY 2$/);
    await expect(page.locator('#telemetry-msg')).toContainText('RESUMED');

    for (const id of ids.slice(1)) {
      await clamp(page, lobes[id]);
      await page.locator('#submit-btn').click();
    }
    await expect(page.locator('#debrief-dialog')).toBeVisible();
    const shareText = (await page.locator('#debrief-grid-text').textContent()) ?? '';
    expect(shareText).toContain('DAILY 2024-01-01');
    expect(shareText).toContain('daily=2024-01-01');
    await expect(page.locator('#debrief-score')).not.toContainText('OFF THE RECORD');

    // Same date, same board; a replay is off the record and earns nothing
    await page.goto('/index.html?daily=2024-01-01');
    await expect(page.locator('.token-tile')).toHaveCount(16);
    expect(boardKey(await readLobes(page))).toBe(boardKey(lobes));
    await expect(page.locator('#telemetry-msg')).toContainText('OFF THE RECORD');
    for (const id of Object.keys(lobes)) {
      await clamp(page, lobes[id]);
      await page.locator('#submit-btn').click();
    }
    await expect(page.locator('#debrief-dialog')).toBeVisible();
    await expect(page.locator('#debrief-score')).toContainText('+0 XP');
    await expect(page.locator('#debrief-score')).toContainText('OFF THE RECORD');
    // A replay may only review a field note that is already declassified, and its report claims no XP
    await expect(page.locator('#debrief-field-note')).toContainText('FIELD NOTE 01/20 — REVIEW');
    await expect(page.locator('#debrief-stats-grid')).toContainText('1/20');
    expect((await page.locator('#debrief-grid-text').textContent()) ?? '').not.toContain('XP');

    // A link shared from a time zone that is a calendar day ahead keeps its date instead of being rewritten
    const tomorrow = await page.evaluate(() => {
      const d = new Date();
      d.setDate(d.getDate() + 1);
      return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
    });
    await page.goto(`/index.html?daily=${tomorrow}`);
    await expect(page.locator('#case-title')).toContainText(`DAILY ${tomorrow}`);

    // A surgeon with a record who has not filed today's case boots straight into it
    await page.goto('/index.html');
    await expect(page.locator('#case-title')).toContainText('DAILY');
    await expect(page.locator('.token-tile')).toHaveCount(16);

    // The manual shows the record and exactly one declassified field note
    await page.locator('#help-btn').click();
    await expect(page.locator('#help-stats-grid')).toContainText('1/20');
    await expect(page.locator('#field-notes-list details')).toHaveCount(1);
    await expect(page.locator('#help-rank-line')).toContainText('RANK:');
  });

  test('verifies every dealt board has exactly one solution under the model\'s own logits, with red herrings on HARD', async ({ page }) => {
    await page.goto('/index.html?lite=1&seed=1');
    await expect(page.locator('.token-tile')).toHaveCount(16);
    const audit = await page.evaluate(() => {
      type Audit = { lobes: string[]; words: string[][]; herrings: string[]; ambiguous: boolean };
      const auditFn = (window as unknown as { __ptAudit: (seed: number, grade: string) => Audit }).__ptAudit;
      const out: Record<string, { ambiguous: number; withHerrings: number; malformed: number; lobes: number }> = {};
      for (const grade of ['easy', 'medium', 'hard']) {
        const seenLobes = new Set<string>();
        const row = { ambiguous: 0, withHerrings: 0, malformed: 0, lobes: 0 };
        for (let i = 0; i < 150; i++) {
          const a = auditFn(2000 + i * 7919, grade);
          const flat = a.words.flat();
          if (a.lobes.length !== 4 || flat.length !== 16 || new Set(flat).size !== 16) row.malformed += 1;
          if (a.ambiguous) row.ambiguous += 1;
          if (a.herrings.length) row.withHerrings += 1;
          a.lobes.forEach(l => seenLobes.add(l));
        }
        row.lobes = seenLobes.size;
        out[grade] = row;
      }
      return out;
    });
    for (const grade of ['easy', 'medium', 'hard']) {
      expect(audit[grade].malformed).toBe(0);
      expect(audit[grade].ambiguous).toBe(0);
      expect(audit[grade].lobes).toBeGreaterThanOrEqual(100);
    }
    // HARD boards plant decoys far more often than EASY ones
    expect(audit.hard.withHerrings).toBeGreaterThan(audit.easy.withHerrings);
    expect(audit.hard.withHerrings).toBeGreaterThanOrEqual(75);
  });

  test('opens Live Lobotomy Lab and Surgical Manual dialogs cleanly with properly bounded footer buttons on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/index.html?lite=1&seed=1');

    await expect(page.locator('.token-tile')).toHaveCount(16);
    await page.locator('#lab-btn').click();
    await expect(page.locator('#lab-dialog')).toBeVisible();
    await expect(page.locator('#lab-board-results')).toContainText('Amputate a cortical lobe');
    await page.locator('#lab-close-btn').click();
    await expect(page.locator('#lab-dialog')).toBeHidden();

    await page.locator('#help-btn').click();
    await expect(page.locator('#help-dialog')).toBeVisible();

    // Verify #wipe-save-btn and #help-done-btn stay inside #help-dialog and do not overflow their button boxes on mobile
    const helpButtonsFit = await page.evaluate(() => {
      const dlg = document.querySelector('#help-dialog')?.getBoundingClientRect();
      const wipeBtn = document.querySelector('#wipe-save-btn') as HTMLElement | null;
      const doneBtn = document.querySelector('#help-done-btn') as HTMLElement | null;
      if (!dlg || !wipeBtn || !doneBtn) return false;
      const wipeRect = wipeBtn.getBoundingClientRect();
      const doneRect = doneBtn.getBoundingClientRect();
      return (
        wipeRect.width >= 100 &&
        doneRect.width >= 100 &&
        wipeBtn.scrollWidth <= wipeBtn.clientWidth + 2 &&
        doneBtn.scrollWidth <= doneBtn.clientWidth + 2 &&
        wipeRect.left >= dlg.left &&
        wipeRect.right <= dlg.right &&
        doneRect.left >= dlg.left &&
        doneRect.right <= dlg.right
      );
    });
    expect(helpButtonsFit).toBe(true);

    await page.locator('#seed-input').fill('42');
    await page.locator('#seed-load-btn').click();
    await expect(page.locator('#seed-btn')).toContainText('SEED: #0042');
  });

  test('loads the on-device ALBERT-v2 SLM in Live Mode and runs real WASM forward passes', async ({ page }) => {
    await page.goto('/index.html?seed=1');
    await expect(page.locator('#mode-btn')).toContainText('WARD → LIVE', { timeout: 60_000 });
    // The initial board stays put (no reshuffle); live mode applies from the next case.
    await page.keyboard.press('n');
    await expect(page.locator('#mode-btn')).toHaveAttribute('data-mode', 'live');
    await expect(page.locator('#inference-timing')).toContainText('LIVE SLM');

    // Live path: NEXT PATIENT during the report's opening delay starts an async 4-pass build; the stale report
    // must not open over it while the old case is still nominally on the table
    const liveLobes = await readLobes(page);
    for (const id of Object.keys(liveLobes)) {
      await clamp(page, liveLobes[id]);
      await page.locator('#submit-btn').click();
    }
    await page.locator('#next-patient-btn').click();
    await page.waitForTimeout(900);
    await expect(page.locator('#debrief-dialog')).toBeHidden();
    await expect(page.locator('.token-tile')).toHaveCount(16);
    await expect(page.locator('.solved-lobe')).toHaveCount(0);

    // Open Lab and run a live custom sentence forward pass
    await page.locator('#lab-btn').click();
    await page.locator('#lab-input').fill('the best ice cream flavor is [MASK].');
    await page.locator('#lab-run-btn').click();
    await expect(page.locator('#lab-vocab-results .lab-row').first()).toBeVisible();

    // Four identical sentences admit no unique split: the Lab refuses instead of dealing an unfair board
    const defaultProbes: string[] = [];
    for (const id of ['#custom-p1', '#custom-p2', '#custom-p3', '#custom-p4']) {
      defaultProbes.push(await page.locator(id).inputValue());
      await page.locator(id).fill('the best ice cream flavor is [MASK].');
    }
    await page.locator('#synthesize-custom-btn').click();
    await expect(page.locator('#lab-board-results')).toContainText('No board dealt');
    await expect(page.locator('#lab-dialog')).toBeVisible();
    await expect(page.locator('#case-title')).not.toContainText('CUSTOM VIVISECTION');
    for (const [i, id] of ['#custom-p1', '#custom-p2', '#custom-p3', '#custom-p4'].entries()) {
      await page.locator(id).fill(defaultProbes[i]);
    }

    // Synthesize a live 4x4 board from the 4 custom sentences in the Lab
    await page.locator('#synthesize-custom-btn').click();
    await expect(page.locator('#lab-dialog')).toBeHidden();
    await expect(page.locator('#case-title')).toContainText('CUSTOM VIVISECTION');
    await expect(page.locator('.token-tile')).toHaveCount(16);
  });
});

