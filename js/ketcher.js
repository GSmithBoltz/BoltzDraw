/**
 * ketcher.js
 *
 * Handles all Ketcher iframe communication and interaction detection.
 *
 * Ghost-molecule fix
 * ------------------
 * When a template tool (e.g. cyclopropane) is selected and the user hovers
 * over the canvas WITHOUT clicking, Ketcher renders a ghost/preview of that
 * structure internally - and getSmiles() returns it as if it were real.
 * We block all SMILES polling while isHoveringKetcherFrame is true so those
 * ghost structures never reach the properties panel.
 */

let ketcherInstance = null;
let lastCommittedSmiles = '';
let isHoveringKetcherFrame = false;
let userHasInteracted = false;
let updateTimer = null;

// Callback fired whenever the canvas SMILES genuinely changes
let onSmilesChangeFn = null;

/**
 * Initialise Ketcher and start listening for structure changes.
 * @param {Function} onSmilesChange - called with the new SMILES string
 *                                    whenever a structure is committed
 */
export function initKetcher(onSmilesChange) {
    onSmilesChangeFn = onSmilesChange;
    const frame = document.getElementById('ketcher-frame');

    // Path 1 - Ketcher posts an 'init' message when it's ready
    window.addEventListener('message', (event) => {
        if (event.data.eventType === 'init') {
            console.log('Ketcher ready via message');
            ketcherInstance = frame.contentWindow.ketcher;
            _setupInteractionDetection(frame);
            // Signal readiness back to app.js
            window.dispatchEvent(new CustomEvent('ketcherReady'));
        }
    });

    // Path 2 - Direct access fallback after iframe load
    frame.onload = () => {
        setTimeout(() => {
            try {
                if (frame.contentWindow?.ketcher && !ketcherInstance) {
                    ketcherInstance = frame.contentWindow.ketcher;
                    console.log('Ketcher connected directly');
                    _setupInteractionDetection(frame);
                    window.dispatchEvent(new CustomEvent('ketcherReady'));
                }
            } catch (e) {
                console.warn('Direct Ketcher access failed:', e);
            }
        }, 2000);
    };
}

/**
 * Wire up all the mouse/keyboard listeners that drive the ghost-molecule fix
 * and the debounced SMILES change detection.
 */
function _setupInteractionDetection(frame) {

    // --- Ghost-molecule guard ---
    // Set flag ON when cursor enters iframe (ghost may appear)
    frame.addEventListener('mouseenter', () => {
        isHoveringKetcherFrame = true;
        userHasInteracted = false;
    });

    // Set flag OFF when cursor leaves (Ketcher clears ghost - safe to poll)
    frame.addEventListener('mouseleave', () => {
        isHoveringKetcherFrame = false;
        if (userHasInteracted) _triggerDelayedUpdate();
    });

    // mouseup inside frame = structure committed (click places atom/bond, ghost is gone).
    // Safe to force-check even while still hovering — no ghost survives a click.
    document.addEventListener('mouseup', (e) => {
        const rect = frame.getBoundingClientRect();
        const inside = (
            e.clientX >= rect.left && e.clientX <= rect.right &&
            e.clientY >= rect.top  && e.clientY <= rect.bottom
        );
        if (inside) {
            userHasInteracted = true;
            _forceDelayedCheck(300);   // bypass ghost guard — click commits the structure
        }
    });

    // Parent-document keyboard events (Ctrl+Z, Delete, etc. when iframe loses focus)
    document.addEventListener('keyup', (e) => {
        if (['Delete', 'Backspace', 'Enter', 'v', 'V', 'z', 'Z'].includes(e.key)) {
            userHasInteracted = true;
            _triggerDelayedUpdate();
        }
    });

    // Keyboard events INSIDE the iframe (atom replacements: O, N, C, S, …).
    // Ketcher is same-origin so we can attach directly. Keyboard input cannot
    // create ghost molecules (only template-hover can), so bypass ghost guard.
    try {
        frame.contentWindow.document.addEventListener('keyup', () => {
            userHasInteracted = true;
            _forceDelayedCheck(350);   // bypass ghost guard — key commits the change
        });
    } catch (e) {
        console.warn('Could not attach iframe keyup listener:', e);
    }

    // Reduced interval: 400 ms instead of 1 s for a snappier fallback
    setInterval(_checkForCommittedChanges, 400);
}

/**
 * Load a SMILES string directly into the Ketcher editor.
 * Resets the committed-SMILES cache so the normal polling loop will
 * pick up the new structure and update the properties panel within ~1 s.
 */
export async function loadSmiles(smiles) {
    if (!ketcherInstance) throw new Error('Ketcher not ready');
    await ketcherInstance.setMolecule(smiles);
    // Clear cache — next polling tick detects the change and fires onSmilesChangeFn
    lastCommittedSmiles = '';
}

/**
 * Debounce helper - waits 300 ms after last interaction before polling.
 * Skipped if mouse is still inside the frame (ghost could still be live).
 */
function _triggerDelayedUpdate() {
    if (isHoveringKetcherFrame) return;
    if (updateTimer) clearTimeout(updateTimer);
    updateTimer = setTimeout(_checkForCommittedChanges, 300);
}

/**
 * Force-check helper - bypasses the ghost guard.
 * Used after a mouseup or keyup inside the frame, where a click/keystroke
 * commits the structure and no ghost can survive.
 */
function _forceDelayedCheck(delay) {
    if (updateTimer) clearTimeout(updateTimer);
    updateTimer = setTimeout(() => _checkForCommittedChanges(true), delay);
}

/**
 * Read current SMILES from Ketcher. Only proceeds if:
 *   - Ketcher is initialised
 *   - Mouse is outside the frame OR force=true (ghost-molecule guard)
 *   - SMILES has actually changed since last commit
 */
async function _checkForCommittedChanges(force = false) {
    if (!ketcherInstance) return;
    if (!force && isHoveringKetcherFrame) return; // ghost guard

    try {
        const smiles = await ketcherInstance.getSmiles();
        if (smiles !== lastCommittedSmiles) {
            lastCommittedSmiles = smiles;
            if (typeof onSmilesChangeFn === 'function') {
                onSmilesChangeFn(smiles);
            }
        }
    } catch (e) {
        // Silently ignore - Ketcher can throw during tool transitions
    }
}
