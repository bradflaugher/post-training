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

  test('detects One-Away (3/4 synapses aligned) and decrements defibrillator charge on mistake', async ({ page }) => {
    await page.goto('/index.html?lite=1&seed=1');

    // Stimulate 3 correct tokens for Probe 01
    const hintBtn = page.locator('#hint-btn');
    await hintBtn.click();
    await hintBtn.click();
    await hintBtn.click();

    // Pick 1 token from a different lobe
    const nonMatchingTile = page.locator('.token-tile:not([data-lobe-id="ice_cream"])').first();
    await nonMatchingTile.click();

    await page.locator('#submit-btn').click();
    await expect(page.locator('#telemetry-msg')).toContainText('3/4 SYNAPSES ALIGNED');
    await expect(page.locator('.volt-pip.lost')).toHaveCount(1);
  });

  test('completes a full surgical case and shows the Post-Op Autopsy Report modal without colored emoji squares', async ({ page }) => {
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
  });

  test('cycles entanglement difficulty grades and synthesizes billions of procedural seeds', async ({ page }) => {
    await page.goto('/index.html?lite=1&seed=999999');
    await expect(page.locator('#seed-btn')).toContainText('SEED: #999999');
    await expect(page.locator('#grade-btn')).toContainText('G3: CRITICAL');

    await page.locator('#grade-btn').click();
    await expect(page.locator('#grade-btn')).toContainText('G1: CLEAN');
    await expect(page.locator('.token-tile')).toHaveCount(16);

    await page.locator('#grade-btn').click();
    await expect(page.locator('#grade-btn')).toContainText('G2: ENTANGLED');
    await expect(page.locator('.token-tile')).toHaveCount(16);
  });

  test('opens Live Lobotomy Lab and Surgical Manual dialogs cleanly', async ({ page }) => {
    await page.goto('/index.html?lite=1&seed=1');

    await page.locator('#lab-btn').click();
    await expect(page.locator('#lab-dialog')).toBeVisible();
    await expect(page.locator('#lab-board-results')).toContainText('Amputate a cortical lobe');
    await page.locator('#lab-close-btn').click();
    await expect(page.locator('#lab-dialog')).toBeHidden();

    await page.locator('#help-btn').click();
    await expect(page.locator('#help-dialog')).toBeVisible();
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

