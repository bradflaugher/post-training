import { test, expect } from '@playwright/test';

test.describe('POST-TRAINING — Neural Vivisection Word Game', () => {
  test('renders 16 mystery tokens with zero category spoilers and fits mobile viewports without scrolling', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/index.html?lite=1&seed=1');
    await expect(page.locator('#mode-btn')).toHaveAttribute('data-mode', 'ward');
    await expect(page.locator('#case-title')).toContainText('CASE #001');

    // No category or probe spoilers are shown before a category is solved
    await expect(page.locator('.solved-lobe')).toHaveCount(0);
    await expect(page.locator('#active-probe-title')).toContainText('0/4 LOBES DECLASSIFIED');

    const tiles = page.locator('.token-tile');
    await expect(tiles).toHaveCount(16);

    // Unsolved tiles keep probabilities locked until their category is solved
    const firstStatus = await tiles.first().locator('.token-prob-val').textContent();
    expect(firstStatus).toBe('LOCKED');

    // Verify the entire game fits on a small phone screen (375x667) without vertical scrolling or clipping
    const fitsViewport = await page.evaluate(() => {
      return document.documentElement.scrollHeight <= window.innerHeight;
    });
    expect(fitsViewport).toBe(true);
    const footerBox = await page.locator('.footer-nav').boundingBox();
    expect(footerBox).not.toBeNull();
    expect((footerBox?.y ?? 0) + (footerBox?.height ?? 0)).toBeLessThanOrEqual(667);
  });

  test('can select 4 matching tokens, amputate a cortical lobe, and reveal its probe & SLM probabilities', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto('/index.html?lite=1&seed=1');

    // Use STIMULATE (Hint) 4 times to select all 4 tokens of the first unsolved lobe (ice_cream)
    const hintBtn = page.locator('#hint-btn');
    for (let i = 0; i < 4; i++) {
      await hintBtn.click();
    }

    const selectedTiles = page.locator('.token-tile.selected');
    await expect(selectedTiles).toHaveCount(4);

    const submitBtn = page.locator('#submit-btn');
    await expect(submitBtn).toBeEnabled();
    await submitBtn.click();

    // 1 lobe solved, 12 tokens left on grid, and the solved lobe reveals its probe + probabilities + logits
    const solvedLobe = page.locator('.solved-lobe');
    await expect(solvedLobe).toHaveCount(1);
    await expect(solvedLobe.first()).toContainText('the best ice cream flavor is [MASK].');
    await expect(solvedLobe.first()).toContainText('% (z=');
    await expect(page.locator('#active-probe-title')).toContainText('1/4 LOBES DECLASSIFIED');
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

  test('completes a full surgical case and shows the Post-Op Autopsy Report modal', async ({ page }) => {
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
    await expect(page.locator('#debrief-grid-text')).toContainText('mode=ward');
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
    await expect(page.locator('#inference-timing')).toContainText('LIVE LLM SYNTHESIS');

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
