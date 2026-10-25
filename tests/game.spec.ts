import { test, expect } from '@playwright/test';

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

    // Use HINT 4 times to clamp all 4 tokens of the first unsolved lobe (ice_cream)
    const hintBtn = page.locator('#hint-btn');
    for (let i = 0; i < 4; i++) {
      await hintBtn.click();
    }

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

    // Stimulate 3 correct tokens for Probe 01
    const hintBtn = page.locator('#hint-btn');
    await hintBtn.click();
    await hintBtn.click();
    await hintBtn.click();

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

    const hintBtn = page.locator('#hint-btn');
    const submitBtn = page.locator('#submit-btn');

    for (let lobe = 0; lobe < 4; lobe++) {
      for (let k = 0; k < 4; k++) {
        await hintBtn.click();
      }
      await submitBtn.click();
    }

    await expect(page.locator('.solved-lobe')).toHaveCount(4);
    await expect(page.locator('#debrief-dialog')).toBeVisible();
    await expect(page.locator('#debrief-summary')).toContainText('SURGERY SUCCESSFUL');
    await expect(page.locator('#debrief-lobes-list .debrief-lobe-item')).toHaveCount(4);
    const shareText = (await page.locator('#debrief-grid-text').textContent()) ?? '';
    expect(shareText).toContain('mode=ward');
    expect(shareText).not.toMatch(/[🟥🟩🟨🟪]/u);

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

  test('opens Live Lobotomy Lab and Surgical Manual dialogs cleanly with properly bounded footer buttons on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/index.html?lite=1&seed=1');

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
    await expect(page.locator('#status-led')).toHaveAttribute('data-state', 'ready', { timeout: 60_000 });
    await expect(page.locator('#mode-btn')).toHaveAttribute('data-mode', 'live');
    await expect(page.locator('#inference-timing')).toContainText('LIVE SLM');

    // Open Lab and run a live custom sentence forward pass
    await page.locator('#lab-btn').click();
    await page.locator('#lab-input').fill('the best ice cream flavor is [MASK].');
    await page.locator('#lab-run-btn').click();
    await expect(page.locator('#lab-vocab-results .lab-row').first()).toBeVisible();

    // Synthesize a live 4x4 board from the 4 custom sentences in the Lab
    await page.locator('#synthesize-custom-btn').click();
    await expect(page.locator('#lab-dialog')).toBeHidden();
    await expect(page.locator('#case-title')).toContainText('CUSTOM VIVISECTION');
    await expect(page.locator('.token-tile')).toHaveCount(16);
  });
});

