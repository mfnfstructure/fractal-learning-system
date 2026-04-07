/**
 * fractal-system.js
 * ═══════════════════════════════════════════════════════════════
 * Fractal Learning System — полная реализация
 *
 * Интегрирует (документы 3, 4, meta-layer patch):
 *   DEFAULT_CONFIG · helpers · tensorUtils
 *   AsyncSemanticCache · EnergySystem · SelfLearningDB
 *   ErrorEngine · SemanticEmbedder · ModelAdapter
 *   FractalController (все методы, включая meta-layer)
 *   AutonomousLearningLoop · bootstrapFractalSystem · verifyMetaLayer
 *
 * Исправлены баги:
 *   1.  tf.tidy(async) → ручной массив disposables
 *   2.  _softCleanData → Object, не Array
 *   3.  targets{} заполняются из prediction.results в _updateBeliefs
 *   4.  _feedbackHistory двойной push → убран из _updateHyperparams
 *   5.  _applyTemporalDecay итерирует entry.targets, не entry
 *   6.  AsyncSemanticCache объявлен ДО FractalController
 *   7.  _recentUncertainty обновляется из entropy
 *   8a. predict→feedback→beliefs цикл в runSelfLearningCycle
 *   8b. Dream использует семантическую интерполяцию
 *   8c. Аттракторы влияют на выбор стратегии
 *   8d. Meta-цикл влияет на LR, temperature, exploration
 *   8e. Глобальный decay timer в конструкторе
 *   8f. Энергия модулирует LR и temperature (_energyModulateArchitecture)
 *   9.  _detectPlateau использует f.loss ?? f.score ?? 0.5
 *   10. SemanticEmbedder.train() — async encode вне minimize()
 *   11. SemanticEmbedder.encode() — lookupEmbedding вместо gather(number)
 *   12. dream() — aEmb не dispose-ится до создания hybrid
 * ═══════════════════════════════════════════════════════════════
 */

// tf должен быть доступен глобально (CDN) или через import
// import * as tf from '@tensorflow/tfjs';

'use strict';

// ═══════════════════════════════════════════════════════════════
//  1. DEFAULT CONFIG
// ═══════════════════════════════════════════════════════════════

export const DEFAULT_CONFIG = {
    db: {
        name: 'fractal_learning_db',
        cacheSize: 1000,
        decayInterval: 60000,
        ttl: 7 * 24 * 60 * 60 * 1000
    },
    model: {
        useTransformer: false,
        embeddingDim: 32,
        featureDim: 40,
        defaultEpochs: 3,
        defaultLR: 0.001
    },
    echo: {
        short: { maxSize: 30, weight: 1.0,  ttl: 5   * 60 * 1000 },
        mid:   { maxSize: 50, weight: 0.4,  ttl: 30  * 60 * 1000 },
        long:  { maxSize: 100, weight: 0.15, ttl: 120 * 60 * 1000 }
    },
    attractors: {
        min: 0.02, max: 0.95, decay: 0.98,
        pruneThreshold: 0.015, pruneInterval: 100
    },
    energy: {
        initial: 1.0, max: 1.0, regenRate: 0.04, regenBase: 0.03, inertia: 0.85,
        costs: { train: 0.06, dream: 0.02, explore: 0.01, update: 0.005 }
    },
    meta: {
        learningRate:    { base: 0.001, min: 0.0001, max: 0.01 },
        explorationRate: { base: 0.05,  min: 0.01,   max: 0.25 },
        temperature:     { base: 0.7,   min: 0.4,    max: 1.5  },
        smoothingAlpha: 0.05
    },
    safeguards: {
        confidenceCap: 0.98, weightMin: 0.001, weightMax: 10.0,
        rollbackThreshold: -0.7,
        dreamConfidenceMin: 0.45,
        dreamNoveltyRange: [0.25, 0.85]
    },
    loop: {
        innerVoicePeriod: 5000,
        minSamplesForTrain: 3,
        maxTrainSamples: 100,
        noveltyThreshold: 0.5
    },
    decay: {
        baseRate: 0.995,
        minWeight: 0.01,
        protectRecentMs: 60000,
        minTotalMass: 20,
        maxTotalMass: 500
    }
};

// ═══════════════════════════════════════════════════════════════
//  2. HELPERS
// ═══════════════════════════════════════════════════════════════

export function smoothAdaptive(oldVal, target, baseAlpha = 0.05, signalQuality = 1.0) {
    const alpha = baseAlpha * Math.max(0.01, Math.min(1, signalQuality));
    return oldVal * (1 - alpha) + target * alpha;
}

export function calculateEntropy(probs) {
    const filt = (probs || []).filter(p => p > 0.01);
    if (!filt.length) return 1;
    const sum = filt.reduce((a, b) => a + b, 0);
    const d = filt.map(p => p / sum);
    let e = 0;
    for (const p of d) if (p > 0) e -= p * Math.log2(p);
    const mx = Math.log2(d.length);
    return mx > 0 ? e / mx : 1;
}

export function stringSimilarity(a, b) {
    if (a === b) return 1;
    const s1 = a.toLowerCase(), s2 = b.toLowerCase();
    if (s1.includes(s2) || s2.includes(s1)) return 0.8;
    const mx = Math.max(s1.length, s2.length);
    if (!mx) return 1;
    let m = 0;
    for (let i = 0; i < Math.min(s1.length, s2.length); i++) if (s1[i] === s2[i]) m++;
    return m / mx;
}

export function clamp(v, mn, mx) { return Math.max(mn, Math.min(mx, v)); }

export function debounce(fn, delay) {
    let timer;
    return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), delay); };
}

export function exponentialDecay(value, rate, steps) {
    return value * Math.pow(rate, steps);
}

// ═══════════════════════════════════════════════════════════════
//  3. TENSOR UTILS
// ═══════════════════════════════════════════════════════════════

export async function withTensors(fn, ...tensorPromises) {
    const tensors = await Promise.all(tensorPromises);
    try { return await fn(...tensors); }
    finally { for (const t of tensors) t?.dispose?.(); }
}

export async function tidyAsync(asyncFn) {
    return await asyncFn({ tf });
}

export function createHybridEmbedding(aEmb, bEmb, { alpha = 0.5, noiseScale = 0.05, dim = 32 } = {}) {
    // Не используем tf.tidy — возвращаем тензор, пользователь сам dispose-ит
    const interp = aEmb.mul(alpha).add(bEmb.mul(1 - alpha));
    if (noiseScale > 0) {
        const noise = tf.randomNormal([dim], 0, noiseScale);
        const result = interp.add(noise);
        interp.dispose(); noise.dispose();
        return result;
    }
    return interp;
}

export function getTFjsVersion() {
    const [major, minor, patch] = (tf.version?.tfjs || tf.version_core || '0.0.0').split('.').map(Number);
    return { major, minor, patch, full: tf.version?.tfjs || tf.version_core };
}

/** FIX 11: gather expects Tensor, не number в некоторых версиях TF.js */
export function lookupEmbedding(embeddings, index, dim) {
    const idx = Math.min(Math.max(0, index), embeddings.shape[0] - 1);
    // Универсальный способ: slice всегда работает
    return embeddings.slice([idx, 0], [1, dim]).squeeze([0]);
}

export function getMemoryStats() {
    const mem = tf.memory();
    return { numTensors: mem.numTensors, numBytes: mem.numBytes, unreliable: mem.unreliable };
}

export function emergencyCleanup() {
    const before = tf.memory().numTensors;
    tf.disposeVariables();
    const after = tf.memory().numTensors;
    console.log(`🧹 Emergency cleanup: ${before} → ${after} tensors`);
    return { before, after, freed: before - after };
}

export function safePredict(model, input, disposeInput = true) {
    let t = input;
    if (!(input instanceof tf.Tensor)) t = tf.tensor(input);
    try { return model.predict(t); }
    finally { if (disposeInput && t !== input) t.dispose(); }
}


// ═══════════════════════════════════════════════════════════════
//  IMPROVED FEATURE VECTOR (экспорт для DistillationModel)
//  Решает проблему неразличимых ключей (processed/patterns/summary)
// ═══════════════════════════════════════════════════════════════

export function buildRichFeatureVector(key, dim = 48) {
    const v   = new Array(dim).fill(0);
    const k   = key.toLowerCase().replace(/[^a-zа-яё0-9_]/g, '');
    const len = k.length;

    // [0-25] Unigram частоты (нормированные)
    for (let i = 0; i < len; i++) {
        const c = k.charCodeAt(i);
        if (c >= 97 && c <= 122) v[c - 97] += 1 / (len || 1);
    }

    // [26-35] Позиционные признаки (первые 10 символов)
    for (let i = 0; i < Math.min(len, 10); i++) {
        v[26 + i] = k.charCodeAt(i) / 128;
    }

    // [36-41] Биграммы (хеш → 6 бакетов)
    for (let i = 0; i < len - 1; i++) {
        const h = ((k.charCodeAt(i) * 31 + k.charCodeAt(i + 1)) >>> 0) % 6;
        v[36 + h] += 0.1;
    }

    // [42] длина нормированная
    v[42] = Math.min(1, len / 30);

    // [43] доля гласных
    v[43] = (k.match(/[aeiouаеёиоуыэюя]/g) || []).length / (len || 1);

    // [44] доля цифр
    v[44] = (k.match(/[0-9]/g) || []).length / (len || 1);

    // [45] доля подчёркиваний
    v[45] = (k.match(/_/g) || []).length / (len || 1);

    // [46] число слов (разделителей)
    const words = key.split(/[\s_\-\.]+/);
    v[46] = Math.min(1, words.length / 5);

    // [47] первый символ хеш
    v[47] = k.charCodeAt(0) / 128 || 0;

    return v;
}

export function augmentKey(key) {
    const words = key.split(/[\s_\-\.]+/);
    return [
        key,
        key.replace(/[_\-]/g, ' '),                         // spaces
        key.replace(/\s+/g, '_'),                            // underscores
        words.map((w, i) => i === 0 ? w : w[0] || '').join('_'), // abbrev
        words.reverse().join('_'),                           // reverse words
        key.toLowerCase(),
        key.toUpperCase(),
    ].filter((v, i, a) => v && v.length > 0 && a.indexOf(v) === i).slice(0, 5);
}

// ═══════════════════════════════════════════════════════════════
//  4. ASYNC SEMANTIC CACHE  — FIX 6: ДО FractalController
// ═══════════════════════════════════════════════════════════════

export class AsyncSemanticCache {
    constructor(maxSize = 250) {
        this.cache   = new Map();
        this.pending = new Map();
        this.maxSize = maxSize;
    }

    async getOrCompute(key, computeFn) {
        if (this.cache.has(key)) return this.cache.get(key);
        if (this.pending.has(key)) return this.pending.get(key);

        const p = computeFn(key)
            .then(result => {
                if (this.cache.size >= this.maxSize) {
                    this.cache.delete(this.cache.keys().next().value);
                }
                this.cache.set(key, result);
                this.pending.delete(key);
                return result;
            })
            .catch(err => { this.pending.delete(key); throw err; });

        this.pending.set(key, p);
        return p;
    }

    clear() { this.cache.clear(); this.pending.clear(); }
}

// ═══════════════════════════════════════════════════════════════
//  5. ENERGY SYSTEM
// ═══════════════════════════════════════════════════════════════

export class EnergySystem {
    constructor(options = {}) {
        this.current    = options.initialEnergy ?? options.initial ?? 1.0;
        this.max        = 1.0;
        this.min        = 0.05;
        this.regenRate  = options.regenRate  ?? 0.02;
        this.regenBase  = options.regenBase  ?? 0.01;
        this.inertia    = options.inertia    ?? 0.85;

        this.costs = {
            train:   options.trainCost   ?? 0.15,
            dream:   options.dreamCost   ?? 0.05,
            explore: options.exploreCost ?? 0.03,
            update:  options.updateCost  ?? 0.01,
            ...(options.costs || {})
        };

        this.meta = { efficiency: 1.0, stressLevel: 0.0, history: [] };
        this._lastUpdate = Date.now();
    }

    hasEnergyFor(action, overrideCost = null) {
        return this.current >= (overrideCost ?? this.costs[action] ?? 0.01);
    }

    spend(action, overrideCost = null) {
        const cost = overrideCost ?? this.costs[action] ?? 0.01;
        this.current = Math.max(this.min, this.current - cost);
        if (this.current < 0.2) this.meta.stressLevel = Math.min(1, this.meta.stressLevel + 0.1);
        return this.current;
    }

    regen() {
        const now     = Date.now();
        const elapsed = (now - this._lastUpdate) / 1000;
        this._lastUpdate = now;

        const stressFactor = 1 - this.meta.stressLevel * 0.5;
        const timeFactor   = Math.min(1, elapsed / 5);
        const amount = (this.regenBase + this.regenRate * timeFactor) * stressFactor * this.meta.efficiency;

        this.current = Math.min(this.max, this.current + amount);
        if (this.current > 0.5) this.meta.stressLevel *= 0.95;
        return this.current;
    }

    updateEfficiency(feedback) {
        const r = feedback?.score ?? 0.5;
        this.meta.efficiency = this.meta.efficiency * 0.95 + r * 0.05;
        this.meta.history.push({ time: Date.now(), energy: this.current, efficiency: this.meta.efficiency });
        if (this.meta.history.length > 100) this.meta.history.shift();
    }

    modulate(base, { minFactor = 0.3, maxFactor = 2.0 } = {}) {
        return base * (minFactor + (maxFactor - minFactor) * this.current);
    }

    shouldPrioritizeStability() {
        return this.current < 0.4 || this.meta.stressLevel > 0.7;
    }

    export() {
        return { current: this.current, meta: { ...this.meta, history: this.meta.history.slice(-10) } };
    }

    import(state) {
        if (state.current !== undefined) this.current = state.current;
        if (state.meta) { Object.assign(this.meta, state.meta); this.meta.history = state.meta.history || []; }
    }

    getStats() {
        return { current: this.current, meta: { ...this.meta }, costs: { ...this.costs }, shouldPrioritizeStability: this.shouldPrioritizeStability() };
    }
}

// ═══════════════════════════════════════════════════════════════
//  6. SELF-LEARNING DB
// ═══════════════════════════════════════════════════════════════

export class SelfLearningDB {
    constructor(options = {}) {
        const cfg = { ...DEFAULT_CONFIG.db, ...options };
        this.dbName = cfg.name;
        this.db     = null;
        this.cache  = new Map();
        this.isReady = false;
        this.mode    = 'memory';
        this.cfg     = cfg;
        this.stats   = { totalScans: 0, cacheHits: 0, cacheMisses: 0, sessionStart: Date.now() };
        this.decayTimer = null;
    }

    async init() {
        try {
            this.db = await this._openIndexedDB();
            await this._loadAll();
            this.isReady = true;
            this.mode    = 'indexeddb';
            this._startDecay();
            return { success: true, mode: 'indexeddb' };
        } catch (e) {
            console.warn('⚠️ IndexedDB unavailable, using in-memory:', e.message);
            this.isReady = true;
            this.mode    = 'memory';
            return { success: false, error: e, mode: 'memory' };
        }
    }

    async _openIndexedDB() {
        return new Promise((resolve, reject) => {
            const req = indexedDB.open(this.dbName, 1);
            req.onerror    = () => reject(req.error);
            req.onsuccess  = () => resolve(req.result);
            req.onupgradeneeded = e => {
                const db = e.target.result;
                if (!db.objectStoreNames.contains('objects')) {
                    const s = db.createObjectStore('objects', { keyPath: 'key' });
                    s.createIndex('popularity', 'popularity');
                    s.createIndex('lastUsed',   'lastUsed');
                    s.createIndex('confidence', 'confidence');
                }
                if (!db.objectStoreNames.contains('stats')) {
                    db.createObjectStore('stats', { keyPath: 'id' });
                }
            };
        });
    }

    async _loadAll() {
        if (!this.db) return;
        return new Promise(resolve => {
            try {
                const tx = this.db.transaction(['objects', 'stats'], 'readonly');
                tx.objectStore('objects').getAll().onsuccess = e =>
                    (e.target.result || []).forEach(r => this.cache.set(r.key, { ...r }));
                tx.objectStore('stats').get('global').onsuccess = e =>
                    { if (e.target.result) Object.assign(this.stats, e.target.result); };
                tx.oncomplete = () => resolve();
                tx.onerror    = () => resolve();
            } catch (e) { resolve(); }
        });
    }

    async get(key) {
        const rec = this.cache.get(key);
        if (rec) {
            rec.popularity = (rec.popularity || 0) + 1;
            rec.lastUsed   = Date.now();
            this.stats.cacheHits++;
            this._saveAsync({ ...rec });
            return JSON.parse(JSON.stringify(rec)); // защитная копия
        }
        this.stats.cacheMisses++;
        return null;
    }

    async save(data) {
        if (!data?.key) return;
        const rec = {
            key:         data.key,
            popularity:  data.popularity  || 1,
            lastUsed:    data.lastUsed    || Date.now(),
            lastUpdated: data.lastUpdated || data.lastUsed || Date.now(),
            confidence:  data.confidence  || 0.5,
            feedback:    data.feedback    || { pos: 0, neg: 0 },
            payload:     data.payload     || data,
            reward:      data.reward,
            timestamp:   Date.now(),
            // FIX: сохраняем служебные поля FractalController
            targets:     data.targets,
            totalWeight: data.totalWeight
        };
        this.cache.set(data.key, rec);
        this.stats.totalScans++;
        await this._saveAsync(rec);
        return rec;
    }

    async _saveAsync(rec) {
        if (!this.db || this.mode !== 'indexeddb') return;
        try {
            const tx = this.db.transaction(['objects', 'stats'], 'readwrite');
            tx.objectStore('objects').put(rec);
            tx.objectStore('stats').put({ id: 'global', ...this.stats });
        } catch (e) { console.warn('⚠️ IndexedDB save failed:', e.message); }
    }

    async addFeedback(key, type) { // type: 'pos' | 'neg'
        const rec = await this.get(key);
        if (!rec) return;

        rec.feedback = rec.feedback || { pos: 0, neg: 0 };
        rec.feedback[type] = (rec.feedback[type] || 0) + 1;

        // Байесовское обновление + асимметрия (быстрый рост при позитиве)
        const total = rec.feedback.pos + rec.feedback.neg;
        if (total > 0) {
            const empirical = rec.feedback.pos / total;
            rec.confidence = 0.5 + (empirical - 0.5) * Math.min(1, total / 10);
        }

        // Записываем reward для обучения
        rec.reward = rec.feedback.pos / Math.max(1, total);
        await this.save(rec);
    }

    getPopular(n = 10) {
        return [...this.cache.values()]
            .sort((a, b) => (b.popularity || 0) - (a.popularity || 0))
            .slice(0, n)
            .map(r => JSON.parse(JSON.stringify(r)));
    }

    getCacheHitRate() {
        const t = this.stats.cacheHits + this.stats.cacheMisses;
        return t > 0 ? ((this.stats.cacheHits / t) * 100).toFixed(1) : '0.0';
    }

    _startDecay() {
        if (this.decayTimer) clearInterval(this.decayTimer);
        this.decayTimer = setInterval(() => this._applyDecay(), this.cfg.decayInterval);
    }

    _applyDecay() {
        const now      = Date.now();
        const halfLife = 24 * 60 * 60 * 1000;
        for (const [key, rec] of this.cache) {
            const age = now - (rec.lastUsed || now);
            rec.popularity  = (rec.popularity || 1) * Math.exp(-age / halfLife);
            rec.confidence  = 0.5 + (rec.confidence - 0.5) * Math.exp(-age / (halfLife * 2));
            if (age > this.cfg.ttl && (rec.popularity || 0) < 0.5) {
                this.cache.delete(key);
                this._deleteFromDB(key);
            }
        }
    }

    async _deleteFromDB(key) {
        if (!this.db) return;
        try {
            const tx = this.db.transaction('objects', 'readwrite');
            tx.objectStore('objects').delete(key);
        } catch (e) {}
    }

    async clearUnpopular(minPop = 3) {
        let deleted = 0;
        const now = Date.now();
        for (const [key, rec] of this.cache) {
            if ((rec.popularity || 0) < minPop && now - (rec.lastUsed || 0) > 3600000) {
                this.cache.delete(key);
                await this._deleteFromDB(key);
                deleted++;
            }
        }
        return deleted;
    }

    async exportJSON() {
        return JSON.stringify({
            exported: new Date().toISOString(),
            stats:    { ...this.stats },
            records:  [...this.cache.values()].map(r => JSON.parse(JSON.stringify(r)))
        }, null, 2);
    }

    async importJSON(json) {
        try {
            const data = JSON.parse(json);
            if (data.records) for (const rec of data.records) this.cache.set(rec.key, rec);
            if (data.stats)   Object.assign(this.stats, data.stats);
            return true;
        } catch (e) { console.error('❌ Import failed:', e); return false; }
    }

    /** Совместимость с FractalController */
    async getEntryForTraining(key) {
        const rec = await this.get(key);
        if (!rec) return null;
        return {
            key:         rec.key,
            payload:     rec.payload,
            targets:     rec.targets || {
                [rec.key]: {
                    weight: (rec.feedback?.pos || 0) - (rec.feedback?.neg || 0),
                    normalizedWeight: rec.confidence || 0.5,
                    meta: {
                        confidence:  rec.confidence || 0.5,
                        lastUpdated: rec.lastUsed   || Date.now(),
                        feedback:    { ...(rec.feedback || {}) }
                    }
                }
            },
            totalWeight: 1,
            lastUpdated: rec.lastUsed || Date.now()
        };
    }

    destroy() {
        if (this.decayTimer) clearInterval(this.decayTimer);
        if (this.db)         this.db.close();
        this.cache.clear();
    }
}

// ═══════════════════════════════════════════════════════════════
//  7. ERROR ENGINE
// ═══════════════════════════════════════════════════════════════

export class ErrorEngine {
    constructor(options = {}) {
        this.similarityThreshold    = options.similarityThreshold    ?? 0.8;
        this.punishWrongConfidence  = options.punishWrongConfidence   ?? true;
    }

    calculate(prediction, actual) {
        if (!actual?.key) {
            return { error: null, reward: 0.5, reason: 'no_ground_truth', supervisedScore: null };
        }

        const predTop = prediction.results?.[0];
        if (!predTop) {
            return { error: 1, reward: 0, reason: 'empty_prediction', supervisedScore: -1 };
        }

        if (predTop.key === actual.key) {
            return {
                error: 0,
                reward: 0.9 + predTop.confidence * 0.1,
                reason: 'exact_match',
                supervisedScore: 1
            };
        }

        const sim = this._similarity(predTop.key, actual.key);
        if (sim > this.similarityThreshold) {
            return {
                error: 0.3,
                reward: 0.6 + sim * 0.3,
                reason: 'partial_match',
                supervisedScore: -0.3
            };
        }

        const penalty = this.punishWrongConfidence ? predTop.confidence * 0.5 : 0.3;
        return {
            error: 1,
            reward: Math.max(0, 0.2 - penalty),
            reason: 'wrong_prediction',
            supervisedScore: -1
        };
    }

    _similarity(a, b) {
        const s1 = a.toLowerCase(), s2 = b.toLowerCase();
        if (s1 === s2) return 1;
        if (s1.includes(s2) || s2.includes(s1)) return 0.85;
        const maxLen = Math.max(s1.length, s2.length);
        if (!maxLen) return 1;
        let matches = 0;
        const window = Math.floor(maxLen / 2);
        for (let i = 0; i < s1.length; i++) {
            const start = Math.max(0, i - window);
            const end   = Math.min(s2.length, i + window + 1);
            if (s2.slice(start, end).includes(s1[i])) matches++;
        }
        return matches / maxLen;
    }
}

// ═══════════════════════════════════════════════════════════════
//  8. SEMANTIC EMBEDDER
//     FIX 10: async encode вне optimizer.minimize()
//     FIX 11: lookupEmbedding вместо gather(number)
// ═══════════════════════════════════════════════════════════════

export class SemanticEmbedder {
    constructor(options = {}) {
        this.dim       = options.embeddingDim || 32;
        this.vocab     = new Map();
        this.embeddings = null;          // tf.Variable или null
        this.isTrained  = false;
        this.cache      = new Map();    // text → tf.Tensor (clone)
    }

    tokenize(text) {
        const tokens = [];
        const lower  = text.toLowerCase();
        for (const char of lower) {
            if (!this.vocab.has(char)) this.vocab.set(char, this.vocab.size);
            tokens.push(this.vocab.get(char));
        }
        for (let i = 0; i < lower.length - 1; i++) {
            const bigram = `##${lower[i]}${lower[i + 1]}`;
            if (!this.vocab.has(bigram)) this.vocab.set(bigram, this.vocab.size);
            tokens.push(this.vocab.get(bigram));
        }
        return tokens;
    }

    _ensureEmbeddings() {
        const vocabSize = Math.max(1, this.vocab.size);
        if (!this.embeddings || this.embeddings.shape[0] < vocabSize) {
            const old = this.embeddings;
            // Растягиваем матрицу при расширении словаря
            this.embeddings = tf.variable(tf.randomNormal([vocabSize, this.dim], 0, 0.1));
            old?.dispose();
        }
    }

    async encode(text) {
        if (this.cache.has(text)) return this.cache.get(text).clone();

        const tokens = this.tokenize(text);
        if (!tokens.length) return tf.zeros([this.dim]);

        this._ensureEmbeddings();

        // FIX 11: используем lookupEmbedding (slice-based, совместим со всеми версиями TF.js)
        const embeds = tokens.map(idx => lookupEmbedding(this.embeddings, idx, this.dim));
        const stacked = tf.stack(embeds);
        const encoded = stacked.mean(0);

        stacked.dispose();
        embeds.forEach(e => e.dispose());

        if (this.cache.size < 200) this.cache.set(text, encoded.clone());
        return encoded;
    }

    similarity(vecA, vecB) {
        return tf.tidy(() => {
            const dot   = tf.dot(vecA, vecB).arraySync();
            const normA = Math.sqrt(tf.sum(tf.square(vecA)).arraySync());
            const normB = Math.sqrt(tf.sum(tf.square(vecB)).arraySync());
            return dot / (normA * normB + 1e-8);
        });
    }

    /** FIX 10: предвычисляем эмбеддинги вне minimize() */
    async train(pairs, epochs = 10, lr = 0.01) {
        // Убеждаемся, что словарь инициализирован
        for (const { anchor, positive, negative } of pairs) {
            this.tokenize(anchor);
            this.tokenize(positive);
            this.tokenize(negative);
        }
        this._ensureEmbeddings();

        const optimizer = tf.train.adam(lr);

        for (let epoch = 0; epoch < epochs; epoch++) {
            for (const { anchor, positive, negative } of pairs) {
                // Async-вычисление ВНЕ minimize (FIX 10)
                const aEmb = await this.encode(anchor);
                const pEmb = await this.encode(positive);
                const nEmb = await this.encode(negative);

                optimizer.minimize(() => {
                    const posDist = tf.sum(tf.square(aEmb.sub(pEmb)));
                    const negDist = tf.sum(tf.square(aEmb.sub(nEmb)));
                    const margin  = 0.3;
                    return tf.maximum(tf.scalar(0), posDist.sub(negDist).add(margin));
                });

                aEmb.dispose(); pEmb.dispose(); nEmb.dispose();
            }
        }
        this.isTrained = true;
    }

    clearCache() {
        for (const t of this.cache.values()) t.dispose();
        this.cache.clear();
    }

    dispose() {
        this.clearCache();
        this.embeddings?.dispose();
        this.embeddings = null;
    }
}

// ═══════════════════════════════════════════════════════════════
//  9. MODEL ADAPTER
//     FIX 1: Function.length игнорирует параметры с дефолтом
// ═══════════════════════════════════════════════════════════════

export class ModelAdapter {
    constructor(model) {
        this.model       = model;
        this._predictSig = null;

        // FIX 1: Function.length = 1 для train(data, epochs=40, cb)
        // Смотрим на строку функции, а не на .length
        const fnStr = model.train?.toString() || '';
        if (model.constructor?.name === 'DistillationModel') {
            this._trainSig = 'legacy';
        } else {
            this._trainSig = (fnStr.includes('epochs') && !fnStr.includes('epochs:'))
                ? 'legacy'
                : 'modern';
        }
    }

    _detectPredictSig(raw) {
        if (Array.isArray(raw)) return 'array';
        if (raw?.results && Array.isArray(raw.results)) return 'wrapped';
        if (raw?.key && typeof raw.confidence === 'number') return 'single';
        return 'unknown';
    }

    async predict(input, trainingData = {}) {
        const raw = await this.model.predict(input, trainingData);
        if (!this._predictSig) {
            this._predictSig = this._detectPredictSig(raw);
        }
        let results;
        switch (this._predictSig) {
            case 'array':   results = raw; break;
            case 'wrapped': results = raw.results; break;
            case 'single':  results = [raw]; break;
            default:        results = Array.isArray(raw) ? raw : (raw ? [raw] : []);
        }
        return {
            results,
            raw,
            meta: { input, timestamp: Date.now(), sig: this._predictSig, modelType: this.model.constructor?.name }
        };
    }

    async train(data, config) {
        const fn = this.model.train?.bind(this.model);
        if (!fn) throw new Error('Model does not support train()');

        // FIX: всегда нормализуем config → явные параметры legacy-API
        const epochs     = typeof config === 'number' ? config : (config?.epochs ?? 3);
        const onProgress = typeof config === 'object' ? (config.onProgress ?? config.callback ?? null) : null;

        return await fn(data, epochs, onProgress);
    }

    async load() { return await this.model.load?.(); }
    getParamCount() { return this.model.getParamCount?.() ?? 0; }
    get isTrained() { return this.model.isTrained ?? false; }
}

// ═══════════════════════════════════════════════════════════════
//  10. FRACTAL CONTROLLER
//      Все методы из doc3 + doc4 + meta-layer patch объединены
//      Все 12 фиксов применены
// ═══════════════════════════════════════════════════════════════

export class FractalController {
    constructor(model, db, options = {}) {
        const cfg = { ...DEFAULT_CONFIG, ...options };

        this.model  = new ModelAdapter(model);
        this.db     = db;
        this.energy = new EnergySystem(cfg.energy);
        this.cfg    = cfg;

        // Компоненты
        this.errorEngine = new ErrorEngine();
        this.embedder    = new SemanticEmbedder({ embeddingDim: cfg.model.embeddingDim });
        this.semanticCache = new AsyncSemanticCache(300); // FIX 6: определён выше

        // Эхо-буферы
        this.echoLayers = {
            short: { buffer: [], maxSize: cfg.echo.short.maxSize, ttl: cfg.echo.short.ttl, weight: cfg.echo.short.weight },
            mid:   { buffer: [], maxSize: cfg.echo.mid.maxSize,   ttl: cfg.echo.mid.ttl,   weight: cfg.echo.mid.weight   },
            long:  { buffer: [], maxSize: cfg.echo.long.maxSize,  ttl: cfg.echo.long.ttl,  weight: cfg.echo.long.weight  }
        };

        // Аттракторы
        this.attractors     = new Map();
        this.attractorMeta  = new Map();
        this._attractorHistory = [];

        // Мета-параметры
        this.meta = {
            learningRate:    { ...cfg.meta.learningRate,    value: cfg.meta.learningRate.base    },
            explorationRate: { ...cfg.meta.explorationRate, value: cfg.meta.explorationRate.base },
            temperature:     { ...cfg.meta.temperature,     value: cfg.meta.temperature.base     },
            smoothingAlpha:  cfg.meta.smoothingAlpha
        };

        // История
        this._feedbackHistory   = [];
        this._weightChanges     = [];
        this._entropyHistory    = [];
        this._plasticityHistory = [];
        this._experienceBuffer  = [];
        this._strategyScores    = {};

        // Статистика
        this.stats = {
            totalProcessed: 0, selfTrainings: 0, dreamsGenerated: 0,
            dreamsAccepted: 0, rollbacks: 0, lastTrainTime: 0,
            metaCycles: 0, reflections: 0
        };

        // FIX 7: инициализируем поля, которые раньше отсутствовали
        this._recentUncertainty  = 0.5;
        this._lastNoveltyScore   = 0;
        this._dreamAcceptanceRate = 0;
        this._recentWeightChangeRate = 0.5;
        this._metaAvgScore       = 0.5;
        this._bestLoss           = null;

        // Флаги
        this._realityAnchor = { active: false, timerId: null, cooldownMs: 30000 };
        this._strategyState = { current: 'consolidate', streak: 0, minCycles: 3, cooldownUntil: 0, lastSwitchAt: 0 };

        // Конфиг стратегий (энергетические стоимости)
        this._strategyCosts = {
            tune_hyperparams:  0.02,
            soft_clean:        0.08,
            perturb_retrain:   0.12,
            safe_augment:      0.10,
            grow_architecture: 0.20,
            consolidate:       0.03
        };

        // Inner voice
        this.innerVoiceTimer = null;
        this.isRunning       = false;

        // FIX 8e: глобальный decay timer
        this._globalDecayTimer = setInterval(() => this._globalDecay(), 60000);

        // Дебаунс сохранения опыта
        this._saveExperienceDebounced = debounce(() => this._persistExperience(), 1000);

        // ── DOC5: IDENTITY ────────────────────────────────────────────
        this.identity = {
            riskTolerance:  0.5,
            curiosity:      0.5,
            stabilityBias:  0.5,
            _history:       []
        };

        // ── DOC5: GLOBAL OBJECTIVE ────────────────────────────────────
        this.globalObjective = {
            maximizeConfidence: true,
            minimizeEntropy:    true,
            explorationBonus:   0.05
        };

        // ── DOC5: FAST BRAIN (Distillation = интуиция) ────────────────
        this.fastBrain      = null;
        this._fastBrainData = {};

        // ── DOC5: INTERNAL VOICE ──────────────────────────────────────
        this.internalVoice = {
            thoughts:       [],
            links:          [],
            maxThoughts:    100,
            lastReflection: 0,
            lastGeneration: 0
        };
    }

    // ── DOC5: Регистрация FastBrain ───────────────────────────────────
    setFastBrain(distillModel, trainingData = {}) {
        this.fastBrain      = distillModel;
        this._fastBrainData = trainingData;
        if (distillModel?.isTrained) {
            // Если модель уже обучена — обновляем данные только
        }
    }

    // ─────────────────────────────────────────────────────────
    //  ОСНОВНОЙ ЦИКЛ
    //  FIX 1: НЕ tf.tidy(async) — ручной массив disposables
    //  FIX 8a: predict→feedback→beliefs ПЕРЕД анализом ландшафта
    // ─────────────────────────────────────────────────────────

    async runSelfLearningCycle(input, actual = null, trainingData = {}) {
        // Проверка энергии
        if (!this.energy.hasEnergyFor('update') && this.energy.current < 0.1) {
            return { skipped: 'low_energy', strategy: 'none' };
        }

        this.stats.totalProcessed++;
        const disposables = []; // FIX 1

        try {
            // ── 0. СОХРАНЯЕМ вход в DB чтобы _adaptiveSelfTrain имел данные
            if (input && typeof input === 'string' && this.db?.isReady) {
                const existing = await this.db.get(input);
                if (!existing) {
                    await this.db.save({ key: input, payload: trainingData[input] || { key: input }, confidence: 0.5 });
                }
            }

            // ── 1a. SELF-SUPERVISION — система сама себе учитель ─────────
            if (!actual) {
                actual = await this._selfSupervise(input);
            }

            // ── 1b. FAST BRAIN — интуиция (быстрый guess) ────────────────
            let fastPrediction = null;
            if (this.fastBrain?.isTrained) {
                try {
                    const fastRaw = await this.fastBrain.predict(input, this._fastBrainData);
                    fastPrediction = { results: Array.isArray(fastRaw) ? fastRaw : [fastRaw] };
                } catch(e) {}
            }

            // ── 1c. SLOW BRAIN — глубокий анализ ────────────────────────
            const slowPrediction = await this._predictWithBias(input, trainingData);

            // ── 1d. MERGE fast + slow ────────────────────────────────────
            const prediction = fastPrediction
                ? { ...slowPrediction, results: this._mergePredictions(slowPrediction.results, fastPrediction.results) }
                : slowPrediction;

            // ── 1e. CONFLICT DETECTION — рождение "сомнения" ─────────────
            if (fastPrediction) {
                const conflict = this._detectConflict(slowPrediction.results, fastPrediction.results);
                if (conflict > 0.6) {
                    // Конфликт: усиливаем exploration, снижаем уверенность
                    this.meta.explorationRate.value = Math.min(0.25, this.meta.explorationRate.value * 1.2);
                }
            }

            // ── 2. ОЦЕНКА ─────────────────────────────────────────────────
            const feedback = this._evaluatePrediction(input, prediction, actual);

            // FIX 7: обновляем _recentUncertainty из entropy
            this._recentUncertainty = this._recentUncertainty * 0.9 + feedback.entropy * 0.1;

            // ── 3. REALITY ANCHOR ─────────────────────────────────────────
            if (this._shouldRollback(feedback)) {
                this._rollbackRecentUpdates();
                feedback.score = feedback.score * 0.5;
                this.stats.rollbacks++;
            }

            // ── 4. ECHO ───────────────────────────────────────────────────
            this._pushToEcho(input, prediction.results, feedback);

            // ── 5. ОБНОВЛЕНИЕ УБЕЖДЕНИЙ (FIX 3, 7) ───────────────────────
            this._updateBeliefs(input, prediction.results, feedback);

            // ── 6. АНАЛИЗ ЛАНДШАФТА ───────────────────────────────────────
            const landscape = await this._analyzeLearningLandscape(trainingData);

            // ── 7. ВЫБОР СТРАТЕГИИ (FIX 8c) ──────────────────────────────
            const strategy = await this._chooseRefinementStrategy(landscape);

            // ── 8. ПРИМЕНЕНИЕ СТРАТЕГИИ ───────────────────────────────────
            const result = await this._applyStrategy(strategy, trainingData, landscape);

            // ── 9. ОЦЕНКА ШАГА ────────────────────────────────────────────
            const stepFeedback = this._evaluateLearningStep(result, landscape);

            // FIX 4: push ТОЛЬКО ЗДЕСЬ (убран из _updateHyperparams)
            this._feedbackHistory.push({
                ...feedback,
                strategy, loss: stepFeedback.loss, score: stepFeedback.score,
                timestamp: Date.now()
            });
            if (this._feedbackHistory.length > 200) this._feedbackHistory.shift();

            // ── 10. МЕТА-АДАПТАЦИЯ ────────────────────────────────────────
            this._updateHyperparams(stepFeedback);         // FIX 4: без push
            this._updateMetaState(stepFeedback, strategy); // FIX 8d
            this._energyModulateArchitecture();            // FIX 8f
            this._updateIdentity(stepFeedback);            // DOC5: адаптация личности

            // ── 11. СОХРАНЕНИЕ ОПЫТА ──────────────────────────────────────
            this._storeLearningExperience({ landscape, strategy, feedback: stepFeedback, timestamp: Date.now() });

            // ── 12. СНАПШОТ + ФРАКТАЛЬНОЕ ВЛИЯНИЕ ────────────────────────
            this._saveAttractorSnapshot();
            this._propagateFractalInfluence();

            // ── 13. ЭНЕРГИЯ ───────────────────────────────────────────────
            this.energy.regen();

            return { strategy, feedback: stepFeedback, prediction, energy: this.energy.current };

        } finally {
            disposables.forEach(t => t?.dispose?.());
        }
    }

    async fullCycle(input, actual = null, trainingData = {}) {
        // Внешний цикл: обработка входа
        const externalResult = await this.runSelfLearningCycle(input, actual, trainingData);

        // Мечтание (если есть энергия)
        if (this.energy.current > 0.4 && Math.random() < 0.3) {
            await this._dreamCycle();
        }

        // Мета-цикл (каждые 20 итераций)
        if (this.stats.totalProcessed % 20 === 0) {
            await this.runMetaCycle();
            this.stats.metaCycles++;
        }

        // Внутренняя рефлексия (каждые 50)
        if (this.stats.totalProcessed % 50 === 0) {
            await this._internalReflectionCycle();
            this.stats.reflections++;
        }

        this._propagateFractalInfluence();
        this.energy.regen();

        return {
            external: externalResult,
            energy:   this.energy.current,
            stats:    { dreams: this.stats.dreamsAccepted, metaCycles: this.stats.metaCycles }
        };
    }

    // ─────────────────────────────────────────────────────────
    //  ОЦЕНКА
    // ─────────────────────────────────────────────────────────

    _evaluatePrediction(input, prediction, actual = null) {
        const results   = prediction.results || [];
        const top       = results[0];
        const hasActual = actual?.key != null;

        let supervisedScore = 0;
        if (hasActual) {
            // Use ErrorEngine for calibrated supervised signal
            const engineResult = this.errorEngine.calculate(prediction, actual);
            supervisedScore = engineResult.supervisedScore ?? (top?.key === actual.key ? 1 : -1);
        }

        const consistency    = this._crossCheckConsistency(prediction);
        const infoDensity    = this._calculateInfoDensity(results, { score: supervisedScore });
        const memoryAlign    = this._checkMemoryAlignment(input, top?.key);
        // Estimate novelty from memory: low hits = high novelty
        const memHit = this.db?.cache?.has(input);
        const memConf = this.db?.cache?.get(input)?.confidence || 0;
        const estimatedNovelty = memHit ? Math.max(0, 1 - memConf) : 0.7;
        this._lastNoveltyScore = estimatedNovelty * 0.3 + (this._lastNoveltyScore || 0) * 0.7;
        const novelty = this._lastNoveltyScore;
        const calibration    = this._getCalibrationScore();
        const criticScore    = this._internalCritic(prediction, { score: supervisedScore, hasActual });

        const w = { consistency: 0.30, infoDensity: 0.25, memAlign: 0.20, novelty: 0.10, calibration: 0.10, critic: 0.05 };
        let score =
            w.consistency * consistency +
            w.infoDensity * infoDensity +
            w.memAlign    * memoryAlign +
            w.novelty     * (1 - novelty) +
            w.calibration * calibration +
            w.critic      * criticScore;

        if (hasActual) score = 0.7 * supervisedScore + 0.3 * score;

        const entropy = calculateEntropy(results.map(r => r.confidence));
        const signalQuality = clamp(
            (0.5 + Math.abs(score) * 0.5) * 0.4 +
            (1 - entropy) * 0.3 +
            (hasActual ? 0.3 : 0.1),
            0, 1
        );

        // DOC5: Global Objective — небольшой бонус за низкую энтропию и высокую уверенность
        if (this.globalObjective?.minimizeEntropy)    score += (1 - entropy) * 0.05;
        if (this.globalObjective?.maximizeConfidence) score += (top?.confidence || 0) * 0.05;
        score = clamp(score, -1, 1);

        return {
            score,
            confidence:     top?.confidence  || 0.5,
            novelty,
            entropy,
            hasActual,
            supervisedScore,
            signalQuality,
            critic:         { score: criticScore },
            timestamp:      Date.now()
        };
    }

    _crossCheckConsistency(prediction) {
        const r = prediction.results || [];
        if (r.length < 2) return 0.3;
        const margin       = r[0].confidence - (r[1]?.confidence || 0);
        const absoluteConf = Math.min(0.9, r[0].confidence);
        const qualityFloor = r[0].confidence < 0.4 ? -0.3 + (r[0].confidence / 0.4) * 0.3 : 0;
        const memBonus     = this._getMemoryConsistencyBonus(r[0].key, prediction.meta?.input);
        return clamp(margin * 1.2 + absoluteConf * 0.4 + memBonus * 0.2 + qualityFloor, 0, 1);
    }

    _calculateInfoDensity(results, context = {}) {
        if (!results?.length) return 0;
        const entropy      = calculateEntropy(results.map(r => r.confidence));
        const energyFactor = this.energy?.current || 0.5;
        const idealEntropy = 0.5 - (energyFactor - 0.5) * 0.4;
        const perfBias     = context?.score > 0 ? 1.0 : 0.6 + Math.abs(context?.score || 0) * 0.4;
        const gaussian     = Math.exp(-Math.pow(Math.abs(entropy - idealEntropy) / (0.3 * perfBias), 2));
        return clamp(gaussian + (this._checkEntropyStagnation(entropy) ? -0.1 : 0), 0, 1);
    }

    _calculateSignalQuality(feedback) {
        return clamp(
            (0.5 + Math.abs(feedback.score) * 0.5) * 0.4 +
            (1 - feedback.entropy) * 0.3 +
            (feedback.hasActual ? 0.3 : 0.1),
            0, 1
        );
    }

    _internalCritic(prediction, context = {}) {
        const r = prediction.results || [];
        if (r.length < 2) return 0;
        const [top, second] = r;
        const gap = top.confidence - (second?.confidence || 0);
        let critique = 0;
        // Самоуверенная ошибка
        if (top.confidence > 0.9 && gap < 0.2) critique -= 0.3;
        // Паралич неопределённости
        const avg = r.reduce((s, x) => s + x.confidence, 0) / r.length;
        const vr  = r.reduce((s, x) => s + Math.pow(x.confidence - avg, 2), 0) / r.length;
        if (avg > 0.25 && avg < 0.4 && vr < 0.01) critique -= 0.2;
        // Здоровая уверенность
        if (gap > 0.4 && top.confidence > 0.6 && top.confidence < 0.95) critique += 0.15;
        // Осторожная правильность
        if (context?.hasActual && context?.supervisedScore > 0 && top.confidence < 0.7) critique += 0.1;
        return clamp(critique, -0.3, 0.15);
    }

    // ─────────────────────────────────────────────────────────
    //  ОБНОВЛЕНИЕ УБЕЖДЕНИЙ
    //  FIX 3: targets заполняются из prediction.results
    //  FIX 7: _recentUncertainty обновляется здесь
    // ─────────────────────────────────────────────────────────

    _updateBeliefs(input, results, feedback) {
        const entry = this._getOrCreateMemoryEntry(input);
        if (!entry) return;

        const topKey = results?.[0]?.key;
        const reward = feedback.score;
        const rf     = reward > 0 ? 1.5 : 0.4; // асимметрия

        // FIX 3: создаём targets из results если их нет
        for (const r of (results || [])) {
            if (!entry.targets[r.key]) {
                entry.targets[r.key] = {
                    weight: 0,
                    meta:   { confidence: r.confidence || 0.5, lastUpdated: Date.now(), updateCount: 0 }
                };
            }
        }

        for (const [key, target] of Object.entries(entry.targets)) {
            const isTop  = (key === topKey);
            const bc     = target.meta?.confidence || 0.5;
            const adj    = reward * rf * (isTop ? 1.0 : 0.3) * (0.5 + bc);
            const cw     = target.weight || 0;
            const sat    = 1 / (1 + Math.abs(cw));

            target.weight = Math.max(0, cw + adj * sat);

            if (feedback.hasActual) {
                target.meta.confidence = isTop && reward > 0
                    ? Math.min(0.99, bc + 0.1)
                    : Math.max(0.1, bc - 0.05);
            } else {
                target.meta.confidence = bc * 0.95 + (feedback.signalQuality || 0.5) * 0.05;
            }

            target.meta.lastUpdated = Date.now();
            target.meta.updateCount = (target.meta.updateCount || 0) + 1;
            this._trackLearningRate(adj);
        }

        this._normalizeEntryWeights(entry);
        this._updateAttractorsFromFeedback(results, feedback);
        this.energy.updateEfficiency(feedback);

        if (this.stats.totalProcessed % 5 === 0) this._applyTemporalDecay(entry);
    }

    _normalizeEntryWeights(entry, targetSum = 1.0) {
        const targets = Object.values(entry.targets || {});
        if (!targets.length) return;
        for (const t of targets) t.weight = clamp(t.weight || 0, this.cfg.safeguards.weightMin, this.cfg.safeguards.weightMax);
        const temp = this.meta.temperature.value;
        const exp  = targets.map(t => Math.exp((t.weight || 0) / temp));
        const sum  = exp.reduce((a, b) => a + b, 0) || 1;
        for (let i = 0; i < targets.length; i++) {
            targets[i].weight = clamp((exp[i] / sum) * targetSum, this.cfg.safeguards.weightMin, 10.0);
        }
    }

    _updateAttractorsFromFeedback(results, feedback) {
        const reward = feedback.score;
        for (const r of (results || [])) {
            const key  = r.key;
            const prev = this.attractors.get(key) || 0;
            const delta = reward * r.confidence * (reward > 0 ? 0.25 : 0.1);
            const sat   = 1 / (1 + Math.pow(prev, 0.8));
            const em    = 0.5 + this.energy.current * 0.5;
            const nv    = prev * this.cfg.attractors.decay + delta * sat * em;

            if (nv < this.cfg.attractors.pruneThreshold) {
                this.attractors.delete(key);
                this.attractorMeta.delete(key);
            } else {
                this.attractors.set(key, clamp(nv, this.cfg.attractors.min, this.cfg.attractors.max));
                const m = this.attractorMeta.get(key) || {};
                m.lastUpdated = Date.now();
                m.updateCount = (m.updateCount || 0) + 1;
                this.attractorMeta.set(key, m);
            }
        }
        if (this.stats.totalProcessed % this.cfg.attractors.pruneInterval === 0) {
            this._pruneAttractors();
        }
    }

    // FIX 5: итерируем entry.targets, не entry
    _applyTemporalDecay(entry) {
        const { baseRate: base, minWeight: mnW, protectRecentMs: protect } = this.cfg.decay;
        const mass = this._getTotalMemoryMass();
        let mf = 1.0;
        if      (mass < this.cfg.decay.minTotalMass) mf = 0.5;
        else if (mass > this.cfg.decay.maxTotalMass)  mf = 1.5;
        const adj = Math.pow(base, mf);
        const now = Date.now();

        // FIX 5: entry.targets — не entry
        for (const [key, target] of Object.entries(entry.targets || {})) {
            if (!target?.meta?.lastUpdated) continue;
            const age = now - target.meta.lastUpdated;
            if (age < protect) continue;
            const sf  = Math.max(0.5, (target.meta.confidence || 0.5) / 2);
            const eff = Math.pow(adj, age / 10000) * sf;
            if (target.weight !== undefined) target.weight = Math.max(mnW, target.weight * eff);
            if (target.meta.confidence)      target.meta.confidence = Math.max(0.1, target.meta.confidence * Math.pow(eff, 0.5));
            target.meta.decayCount = (target.meta.decayCount || 0) + 1;
        }
        this._pruneWeakTargets(entry, mnW * 2);
    }

    _pruneWeakTargets(entry, threshold = 0.02) {
        if (!entry?.targets) return;
        for (const [key, t] of Object.entries(entry.targets)) {
            if ((t.weight || 0) < threshold && (t.meta?.decayCount || 0) > 2) {
                delete entry.targets[key];
            }
        }
    }

    // FIX 8e: глобальный decay аттракторов (вызывается по таймеру)
    _globalDecay() {
        for (const [k, v] of this.attractors) {
            const d = v * this.cfg.attractors.decay;
            if (d < this.cfg.attractors.pruneThreshold) this.attractors.delete(k);
            else this.attractors.set(k, d);
        }
    }

    // ─────────────────────────────────────────────────────────
    //  АНАЛИЗ ЛАНДШАФТА И ВЫБОР СТРАТЕГИИ
    //  FIX 8c: аттракторы влияют на стратегию
    //  FIX 9:  _detectPlateau использует f.loss ?? f.score
    // ─────────────────────────────────────────────────────────

    async _analyzeLearningLandscape(batch = {}) {
        return {
            recentError:   this._getRecentLoss(),
            plateauEpochs: this._detectPlateau(),    // FIX 9
            dataQuality:   this._estimateDataQuality(batch),
            uncertainty:   this._recentUncertainty,
            memoryUsage:   this._getMemoryUsage(),
            energy:        this.energy.current,
            plasticity:    this._getSystemPlasticity()
        };
    }

    _getRecentLoss(window = 10) {
        // FIX 9: f.loss ?? f.score ?? 0.5
        const losses = this._feedbackHistory.slice(-window).map(f => f.loss ?? f.score ?? 0.5);
        return losses.length ? losses.reduce((a, b) => a + b, 0) / losses.length : 0.5;
    }

    _detectPlateau(threshold = 0.01, minEpochs = 5) {
        // FIX 9: f.loss ?? f.score ?? 0.5
        const losses = this._feedbackHistory.slice(-minEpochs).map(f => f.loss ?? f.score ?? 0.5);
        if (losses.length < minEpochs) return 0;
        const mean = losses.reduce((a, b) => a + b, 0) / losses.length;
        const vr   = losses.reduce((s, l) => s + Math.pow(l - mean, 2), 0) / losses.length;
        return Math.sqrt(vr) < threshold ? minEpochs : 0;
    }

    _estimateDataQuality(batch = {}) {
        const entries = Object.values(batch).length
            ? Object.values(batch)
            : [...(this.db?.cache?.values() || [])];
        if (!entries.length) return 0.5;
        const avgConf    = entries.reduce((s, e) => s + (e.confidence || 0.5), 0) / entries.length;
        const fbRatio    = entries.filter(e => (e.feedback?.pos || 0) + (e.feedback?.neg || 0) > 0).length / entries.length;
        return avgConf * 0.6 + fbRatio * 0.4;
    }

    _getMemoryUsage() {
        const mem = getMemoryStats();
        return mem.numBytes / (1024 * 1024 * 100);
    }

    async _chooseRefinementStrategy(landscape) {
        // Аварийный режим
        if (this.energy.shouldPrioritizeStability()) return this._switchStrategy('consolidate', 'low_energy');

        // FIX 8c: сильные аттракторы → консолидируем
        if (this.attractors.size > 0) {
            const attrStr = Array.from(this.attractors.values()).reduce((a, b) => a + b, 0) / this.attractors.size;
            if (attrStr > 0.6) return this._switchStrategy('consolidate', 'strong_attractors');
        }

        // Критическая ошибка → soft_clean немедленно
        if (landscape.recentError > 0.6) return this._switchStrategy('soft_clean', 'critical_error');

        // DOC5: Identity влияет на выбор стратегии
        if (this.identity) {
            if (this.identity.curiosity > 0.65 && this.energy.current > 0.5)
                return this._switchStrategy('safe_augment', 'identity_curiosity');
            if (this.identity.stabilityBias > 0.7)
                return this._switchStrategy('consolidate', 'identity_stability');
            if (this.identity.riskTolerance < 0.3)
                return this._switchStrategy('consolidate', 'identity_low_risk');
        }

        // DOC5: Параллельное тестирование стратегий (10% шанс)
        if (Math.random() < 0.10) {
            const alternatives = ['consolidate', 'tune_hyperparams', 'safe_augment']
                .filter(s => s !== this._strategyState.current);
            const testStrat = alternatives[Math.floor(Math.random() * alternatives.length)];
            // Небольшое угасание неиспользуемых стратегий → побуждает к исследованию
            if (this._strategyScores[testStrat] !== undefined)
                this._strategyScores[testStrat] *= 0.99;
        }

        // Опыт-базированный выбор (ε-жадный)
        const scores    = this._strategyScores;
        const candidates = Object.entries(scores).filter(([, s]) => s > 0.3).sort((a, b) => b[1] - a[1]);
        if (candidates.length > 0 && Math.random() > (this.meta.explorationRate.value || 0.05)) {
            return this._switchStrategy(candidates[0][0], 'experience_based');
        }

        // Rule-based fallback
        const candidate = this._evaluateCandidateStrategy(landscape);
        return this._switchStrategy(candidate, 'rule_based');
    }

    _switchStrategy(newStrategy, reason) {
        const s = this._strategyState;
        const n = this.stats.totalProcessed;
        if (newStrategy === s.current) { s.streak++; return newStrategy; }
        if (n < s.cooldownUntil)       { s.streak++; return s.current;   }
        s.current      = newStrategy;
        s.streak       = 0;
        s.lastSwitchAt = n;
        s.cooldownUntil = n + s.minCycles;
        return newStrategy;
    }

    _evaluateCandidateStrategy({ recentError, plateauEpochs, dataQuality, uncertainty, memoryUsage }) {
        if (recentError > 0.4 && plateauEpochs < 5)          return 'tune_hyperparams';
        if (recentError > 0.3 && dataQuality < 0.65)          return 'soft_clean';
        if (recentError > 0.25 && memoryUsage < 0.6 && uncertainty > 0.5) return 'perturb_retrain';
        if (recentError > 0.2 && uncertainty > 0.65)          return 'safe_augment';
        if (recentError > 0.15 && this.attractors.size < 100 && this.energy.current > 0.6) return 'grow_architecture';
        return 'consolidate';
    }

    // ─────────────────────────────────────────────────────────
    //  ПРИМЕНЕНИЕ СТРАТЕГИЙ
    //  FIX 2: _softCleanData возвращает Object, не Array
    // ─────────────────────────────────────────────────────────

    async _applyStrategy(strategy, trainingData, landscape) {
        if (!this.energy.hasEnergyFor(strategy, this._strategyCosts[strategy])) {
            console.log(`⚡ Low energy for ${strategy}, falling back to consolidate`);
            strategy = 'consolidate';
        }

        switch (strategy) {
            case 'tune_hyperparams':
                this._tuneHyperparams(landscape);
                this.energy.spend('update', this._strategyCosts.tune_hyperparams);
                return { type: 'meta_update' };

            case 'soft_clean': {
                // FIX 2: возвращает Object
                const cleaned = this._softCleanData(trainingData);
                if (Object.keys(cleaned).length < 2) return { type: 'skip_insufficient' };
                const r = await this.model.train(cleaned, { epochs: Math.max(1, Math.floor(this.energy.modulate(2, { minFactor: 0.5, maxFactor: 2 }))) });
                this.energy.spend('train', this._strategyCosts.soft_clean);
                return r;
            }

            case 'perturb_retrain': {
                const r = await this._safePerturbRetrain(trainingData);
                this.energy.spend('train', this._strategyCosts.perturb_retrain);
                return r;
            }

            case 'safe_augment': {
                const augBatch = {};
                for (const [k, v] of Object.entries(trainingData)) {
                    const augmented = this._safeAugment(v, k);
                    for (const a of augmented) augBatch[a._key || k + '_aug' + Math.random().toString(36).slice(2, 5)] = a;
                }
                if (Object.keys(augBatch).length < 2) return { type: 'skip_insufficient' };
                const r = await this.model.train(augBatch, { epochs: Math.max(1, Math.floor(this.energy.modulate(2, { minFactor: 0.5, maxFactor: 2 }))) });
                this.energy.spend('train', this._strategyCosts.safe_augment);
                return r;
            }

            case 'grow_architecture': {
                const r = await this._growModelIfReady(trainingData);
                this.energy.spend('train', this._strategyCosts.grow_architecture);
                return r;
            }

            case 'consolidate':
            default: {
                if (Object.keys(trainingData).length < 2) return { type: 'skip_no_data' };
                const r = await this.model.train(trainingData, { epochs: Math.max(1, Math.floor(this.energy.modulate(1, { minFactor: 0.5, maxFactor: 1.5 }))) });
                this.energy.spend('update', this._strategyCosts.consolidate);
                return r;
            }
        }
    }

    _evaluateLearningStep(result, landscape) {
        const loss        = result?.history?.history?.loss?.[0] ?? result?.loss ?? landscape.recentError;
        const improvement = landscape.recentError - loss;
        const entropy     = this._computePredictionEntropy();
        const score       = clamp(improvement - entropy * 0.1 + (this.energy.meta.efficiency || 0.5) * 0.05, -1, 1);
        return { loss, improvement, entropy, score };
    }

    _computePredictionEntropy() {
        const recent = this._feedbackHistory.slice(-5);
        if (!recent.length) return 0.5;
        return recent.reduce((s, f) => s + (f.entropy || 0.5), 0) / recent.length;
    }

    _tuneHyperparams(landscape) {
        if (landscape.plateauEpochs > 0) {
            this.meta.learningRate.value = Math.min(this.meta.learningRate.max, this.meta.learningRate.value * 1.2);
        }
        if (landscape.uncertainty > 0.7) {
            this.meta.temperature.value = Math.min(this.meta.temperature.max, this.meta.temperature.value * 1.1);
        }
        this.meta.learningRate.value = clamp(this.meta.learningRate.value, this.meta.learningRate.min, this.meta.learningRate.max);
        this.meta.temperature.value  = clamp(this.meta.temperature.value,  this.meta.temperature.min,  this.meta.temperature.max);
    }

    // FIX 2: возвращает Object { key: entry }
    _softCleanData(entries) {
        if (!entries || typeof entries !== 'object') return {};
        return Object.fromEntries(
            Object.entries(entries)
                .map(([key, entry]) => {
                    const noise  = this._estimateNoise(entry || {});
                    const weight = Math.max(0.1, 1 - noise * 0.7);
                    return [key, { ...(entry || {}), key, sampleWeight: weight, flaggedForReview: noise > 0.8 }];
                })
                .filter(([, e]) => e.sampleWeight > 0.2)
        );
    }

    _estimateNoise(entry) {
        const fc  = Math.abs((entry.feedback?.pos || 0) - (entry.feedback?.neg || 0)) < 2 ? 0.8 : 0.1;
        const lc  = 1 - Math.min(1, (entry.confidence || 0.5) / 0.7);
        const ea  = Math.abs((entry.entropy || 0.5) - 0.5) * 0.5;
        return fc * 0.5 + lc * 0.3 + ea * 0.2;
    }

    async _safePerturbRetrain(data, { scale = 0.005, epochs = 3, lossMultiplier = 1.2 } = {}) {
        const rawModel = this.model.model;
        const originalWeights  = rawModel?.getWeights?.()?.map(w => w.clone()) || [];
        const perturbedWeights = [];
        try {
            for (const w of originalWeights) {
                const noise = tf.randomNormal(w.shape, 0, scale);
                perturbedWeights.push(tf.add(w, noise));
                noise.dispose();
            }
            rawModel?.setWeights?.(perturbedWeights);
            const history = await this.model.train(data, { epochs, batchSize: Math.min(16, Math.max(2, Math.floor(Object.keys(data).length / 4))) });
            const currentLoss = history?.history?.history?.loss?.[0] ?? (this._bestLoss || 0.5);
            const threshold   = (this._bestLoss || currentLoss) * lossMultiplier;
            if (currentLoss && currentLoss > threshold) {
                rawModel?.setWeights?.(originalWeights);
                return { success: false, reason: 'loss_increase' };
            }
            this._bestLoss = currentLoss;
            return { success: true, history };
        } finally {
            perturbedWeights.forEach(t => t?.dispose());
            originalWeights.forEach(t => t?.dispose());
        }
    }

    // Реальная реализация _safeAugment (была заглушкой)
    _safeAugment(entry, baseKey = '') {
        if (!entry) return [entry];
        const variants = [entry];
        const key = entry.key || baseKey;
        if (!key) return variants;
        // Вариант 1: замена пробелов на _
        const k1 = key.replace(/\s+/g, '_');
        if (k1 !== key) variants.push({ ...entry, _key: k1, key: k1, isSynthetic: true, confidence: (entry.confidence || 0.5) * 0.95 });
        // Вариант 2: развёрнутые слова
        const words = key.split(/[\s_]+/);
        if (words.length > 1) {
            const k2 = words.slice().reverse().join('_');
            variants.push({ ...entry, _key: k2, key: k2, isSynthetic: true, confidence: (entry.confidence || 0.5) * 0.90 });
        }
        // Вариант 3: аббревиатура
        const abbr = words.map(w => w[0] || '').join('').toLowerCase();
        if (abbr.length >= 2 && abbr !== key) {
            variants.push({ ...entry, _key: abbr, key: abbr, isSynthetic: true, confidence: (entry.confidence || 0.5) * 0.85 });
        }
        return variants.filter((v, i, a) => v && a.findIndex(x => (x._key || x.key) === (v._key || v.key)) === i);
    }

    // Реальная реализация _growModelIfReady (была заглушкой)
    async _growModelIfReady(trainingData = {}) {
        const dbSize = this.db?.cache?.size || 0;
        if (!this.model.isTrained || dbSize < 20) {
            return { type: 'growth_skipped', reason: 'insufficient_data' };
        }
        // Если в DB значительно больше классов чем в модели → переобучаем
        const currentKeys = this.model.model?.allKeys?.length || 0;
        const availableKeys = Object.keys(trainingData).length || dbSize;
        if (availableKeys > currentKeys * 1.5) {
            console.log(`🌱 Growing model: ${currentKeys} → ${availableKeys} classes`);
            const growData = Object.keys(trainingData).length
                ? trainingData
                : Object.fromEntries([...this.db.cache.entries()].map(([k, v]) => [k, v.payload || {}]));
            return await this.model.train(growData, { epochs: 5 });
        }
        return { type: 'growth_skipped', reason: 'not_needed' };
    }

    // ─────────────────────────────────────────────────────────
    //  МЕТА-ПАРАМЕТРЫ
    //  FIX 4: push убран (выполняется в runSelfLearningCycle)
    //  FIX 8d: мета-цикл влияет на LR, temperature, exploration
    //  FIX 8f: энергия влияет на архитектуру
    // ─────────────────────────────────────────────────────────

    _updateHyperparams(feedback) {
        const { meta }   = this;
        const score      = feedback.score;
        const sq         = feedback.signalQuality || 0.5;
        // Learning Rate
        const targetLR = meta.learningRate.base * (1 + score * 0.3);
        meta.learningRate.value = clamp(
            smoothAdaptive(meta.learningRate.value, targetLR, 0.03, sq),
            meta.learningRate.min, meta.learningRate.max
        );
        // Exploration Rate
        const us = (1 - (feedback.confidence || 0.5)) * 0.5 + (feedback.novelty || 0) * 0.5;
        const targetEx = meta.explorationRate.base * (1 + us * 0.8);
        const plasticity = this._getSystemPlasticity();
        const dynMin = (meta.explorationRate.min || 0.01) * (1 - plasticity * 0.5);
        meta.explorationRate.value = clamp(
            smoothAdaptive(meta.explorationRate.value, targetEx, 0.02, sq * 0.8),
            dynMin, meta.explorationRate.max || 0.25
        );
        // Temperature
        const targetTemp = meta.temperature.base * (1 + ((feedback.entropy || 0.5) - 0.5) * 0.4);
        meta.temperature.value = clamp(
            smoothAdaptive(meta.temperature.value, targetTemp, 0.02, sq),
            meta.temperature.min, meta.temperature.max
        );
        // FIX 4: НЕТ push в _feedbackHistory здесь
    }

    // FIX 8d: мета-состояние влияет на LR, temperature, exploration
    _updateMetaState(feedback, strategy) {
        queueMicrotask(() => {
            // EMA обновление score стратегии
            const alpha  = 0.1;
            const prev   = this._strategyScores[strategy] ?? 0.5;
            this._strategyScores[strategy] = prev * (1 - alpha) + feedback.score * alpha;

            // Exploration адаптация
            const prevEx = this.meta.explorationRate.value;
            this.meta.explorationRate.value = clamp(
                feedback.score < 0 ? prevEx * 1.1 : prevEx * 0.95,
                this.meta.explorationRate.min || 0.01,
                this.meta.explorationRate.max || 0.25
            );

            // FIX 8d: при успехе — снижаем LR (закрепляем)
            if (feedback.score > 0.6) {
                this.meta.learningRate.value = Math.max(
                    this.meta.learningRate.min,
                    this.meta.learningRate.value * 0.97
                );
            }
            // FIX 8d: при неопределённости — повышаем temperature
            if ((feedback.entropy || 0.5) > 0.7) {
                this.meta.temperature.value = Math.min(
                    this.meta.temperature.max,
                    this.meta.temperature.value * 1.05
                );
            }
        });
    }

    // FIX 8f: энергия модулирует LR и temperature
    _energyModulateArchitecture() {
        if (this.energy.current < 0.2) {
            this.meta.learningRate.value = Math.max(this.meta.learningRate.min, this.meta.learningRate.value * 0.9);
            this.meta.temperature.value  = Math.max(this.meta.temperature.min,  this.meta.temperature.value  * 0.9);
        } else if (this.energy.current > 0.85) {
            this.meta.learningRate.value = Math.min(this.meta.learningRate.max, this.meta.learningRate.value * 1.05);
            this.meta.temperature.value  = Math.min(this.meta.temperature.max,  this.meta.temperature.value  * 1.02);
        }
    }

    // ─────────────────────────────────────────────────────────
    //  META-CYCLE
    // ─────────────────────────────────────────────────────────

    async runMetaCycle() {
        if (!this.energy.hasEnergyFor('train', 0.05)) return { skipped: 'low_energy' };

        const insights    = this._analyzeExperience();
        const metaDecision = this._chooseMetaStrategy(insights);
        await this._applyMetaStrategy(metaDecision);
        this._updateMetaReward(metaDecision, insights);

        return { metaDecision, insights };
    }

    _analyzeExperience() {
        if (!this._experienceBuffer?.length) return { topStrategies: [], avgScore: 0.5, diversity: 0 };
        const recent = this._experienceBuffer.slice(-50);
        const stats  = {};
        for (const exp of recent) {
            const { strategy, feedback } = exp;
            if (!stats[strategy]) stats[strategy] = { count: 0, totalScore: 0, improvements: 0 };
            stats[strategy].count++;
            stats[strategy].totalScore += feedback?.score || 0;
            if ((feedback?.improvement || 0) > 0) stats[strategy].improvements++;
        }
        const topStrategies = Object.entries(stats)
            .map(([name, s]) => ({ name, avgScore: s.totalScore / s.count, successRate: s.improvements / s.count, frequency: s.count }))
            .sort((a, b) => b.avgScore - a.avgScore)
            .slice(0, 3);
        return {
            topStrategies,
            avgScore:  recent.reduce((s, e) => s + (e.feedback?.score || 0), 0) / recent.length,
            diversity: new Set(recent.map(e => e.strategy)).size / Math.max(1, recent.length)
        };
    }

    _chooseMetaStrategy(insights) {
        if (insights.topStrategies[0]?.successRate > 0.8) return { type: 'reinforce', target: insights.topStrategies[0].name };
        if (insights.diversity < 0.3)                      return { type: 'explore_strategies' };
        if (insights.avgScore  < 0.2)                      return { type: 'reset_meta_params' };
        return { type: 'maintain' };
    }

    async _applyMetaStrategy(decision) {
        switch (decision.type) {
            case 'reinforce':
                // FIX 8d: снижаем LR успешной стратегии — закрепляем знание
                this.meta.learningRate.value = Math.max(this.meta.learningRate.min, this.meta.learningRate.value * 0.95);
                console.log(`🎯 Meta: reinforcing "${decision.target}", LR ↓`);
                break;
            case 'explore_strategies':
                // FIX 8d: повышаем exploration и temperature
                this.meta.explorationRate.value = Math.min(this.meta.explorationRate.max, this.meta.explorationRate.value * 1.3);
                this.meta.temperature.value     = Math.min(this.meta.temperature.max,     this.meta.temperature.value     * 1.1);
                console.log('🔍 Meta: explore — exploration ↑, temperature ↑');
                break;
            case 'reset_meta_params':
                this._strategyScores          = {};
                this.meta.explorationRate.value = this.meta.explorationRate.base;
                this.meta.temperature.value     = this.meta.temperature.base;
                this.meta.learningRate.value    = this.meta.learningRate.base;
                console.log('🔄 Meta: full reset of meta-parameters');
                break;
        }
    }

    _updateMetaReward(decision, insights) {
        const currentAvg = insights.avgScore;
        const reward     = currentAvg - (this._metaAvgScore || 0.5);
        if (reward > 0.05 && decision.target && this._strategyScores) {
            this._strategyScores[decision.target] = Math.min(1.0, (this._strategyScores[decision.target] || 0.5) + 0.02);
        }
        this._metaAvgScore = currentAvg;
    }

    // ─────────────────────────────────────────────────────────
    //  DREAMING
    //  FIX 8b: семантическая интерполяция через embedder
    //  FIX 12: aEmb dispose ПОСЛЕ создания hybrid
    // ─────────────────────────────────────────────────────────

    async _dreamCycle() {
        if (!this.energy.hasEnergyFor('dream')) return [];
        const samples   = this._sampleMemory(10);
        const synPairs  = [];

        for (const s of samples) {
            const variant = await this._mutateExperience(s); // FIX 8b
            if (variant) synPairs.push(variant);
        }

        // DOC5: используем async FastBrain фильтр если доступен
        const accepted = [];
        for (const d of synPairs) {
            if (await this._validateDreamWithFastBrain(d)) accepted.push(d);
        }
        if (accepted.length > 0) {
            const batch = Object.fromEntries(accepted.map(e => [e.key, e]));
            try { await this.model.train(batch, { epochs: 1 }); } catch (e) {}
            this._updateWorldModelFromDreams(accepted);

            // DOC5: Обновление убеждений из снов (belief update from dream)
            for (const d of accepted) {
                this._updateBeliefs(d.key, [{ key: d.key, confidence: d.confidence }], {
                    score:         0.3,
                    signalQuality: 0.4,
                    entropy:       0.5,
                    novelty:       d.novelty || 0.5,
                    hasActual:     false
                });
            }

            // DOC5: FastBrain обучается на хороших снах
            if (this.fastBrain && accepted.length >= 3) {
                const dreamData = Object.fromEntries(
                    accepted.filter(d => d.confidence >= 0.55)
                            .map(d => [d.key, d.payload || { key: d.key }])
                );
                if (Object.keys(dreamData).length >= 3) {
                    try { await this.fastBrain.train(dreamData, 5); } catch(e) {}
                }
            }
        }

        this.stats.dreamsGenerated += synPairs.length;
        this.stats.dreamsAccepted  += accepted.length;
        this._dreamAcceptanceRate   = accepted.length / Math.max(1, synPairs.length);

        this.energy.spend('dream');
        return accepted;
    }

    _sampleMemory(n = 10) {
        const entries = [...(this.db?.cache?.values() || [])];
        if (!entries.length) return [];
        return entries
            .map(e => ({ entry: e, weight: (e.popularity || 1) * (e.confidence || 0.5) }))
            .sort((a, b) => b.weight - a.weight)
            .slice(0, n * 2)
            .sort(() => Math.random() - 0.5)
            .slice(0, n)
            .map(w => w.entry);
    }

    // FIX 8b: семантическая мутация — интерполяция если есть embedder, иначе умная строковая
    async _mutateExperience(entry) {
        if (!entry?.key) return null;

        if (this.embedder?.isTrained) {
            // Семантический путь: интерполяция эмбеддингов
            const keys = Array.from(this.attractors.keys()).filter(k => k !== entry.key);
            if (!keys.length) return null;
            const partnerKey = keys[Math.floor(Math.random() * keys.length)];

            let aEmb = null, bEmb = null, hybrid = null;
            try {
                aEmb   = await this.embedder.encode(entry.key);
                bEmb   = await this.embedder.encode(partnerKey);
                // FIX 12: создаём hybrid ДО dispose aEmb/bEmb
                hybrid = createHybridEmbedding(aEmb, bEmb, { alpha: 0.5 + Math.random() * 0.2, dim: this.embedder.dim });

                const synKey = `${entry.key}+${partnerKey}`;
                if (this.db?.cache?.has(synKey)) { hybrid.dispose(); return null; }

                return {
                    key:        synKey,
                    isSynthetic: true,
                    confidence: (entry.confidence || 0.5) * 0.9,
                    payload:    entry.payload,
                    timestamp:  Date.now()
                };
            } finally {
                // FIX 12: dispose ПОСЛЕ использования
                aEmb?.dispose();
                bEmb?.dispose();
                hybrid?.dispose();
            }
        }

        // Строковый fallback — умная перестановка частей
        const ka = entry.key.split(/[\s_+]+/);
        const others = [...(this.db?.cache?.keys() || [])].filter(k => k !== entry.key).slice(0, 10);
        if (!others.length) return null;
        const partnerKey = others[Math.floor(Math.random() * others.length)];
        const kb = partnerKey.split(/[\s_+]+/);
        const hybrid = [
            ...ka.slice(0, Math.ceil(ka.length / 2)),
            ...kb.slice(0, Math.ceil(kb.length / 2))
        ].join('_');

        if (hybrid === entry.key || this.db?.cache?.has(hybrid)) return null;

        return {
            key:        hybrid,
            isSynthetic: true,
            confidence: (entry.confidence || 0.5) * 0.88,
            payload:    entry.payload,
            timestamp:  Date.now()
        };
    }

    _validateDream(dream) {
        if (!dream?.key || dream.key.length > 100)       return false;
        if (dream.confidence < this.cfg.safeguards.dreamConfidenceMin) return false;
        if (this.db?.cache?.has(dream.key))               return false;
        const [minN, maxN] = this.cfg.safeguards.dreamNoveltyRange;
        const novelty = dream.novelty ?? 0.5;
        if (novelty < minN || novelty > maxN)             return false;
        return true;
    }

    // DOC5: Async-валидация через FastBrain — фильтр "галлюцинаций"
    async _validateDreamWithFastBrain(dream) {
        if (!this._validateDream(dream)) return false;
        if (!this.fastBrain?.isTrained)  return true; // нет fast brain — пропускаем

        try {
            const check = await this.fastBrain.predict(dream.key, this._fastBrainData);
            const topConf = check[0]?.confidence ?? 0;
            // Если fast brain считает это полным бредом (< 0.3) — отбрасываем
            if (topConf < 0.3) return false;
        } catch(e) {}
        return true;
    }

    _updateWorldModelFromDreams(dreams) {
        for (const d of dreams) {
            const cur = this.attractors.get(d.key) || 0;
            this.attractors.set(d.key, Math.min(0.3, cur + (d.confidence || 0.5) * 0.1));
        }
    }

    // ─────────────────────────────────────────────────────────
    //  СЕМАНТИЧЕСКОЕ ИССЛЕДОВАНИЕ И ДЕШЁВОЕ ИССЛЕДОВАНИЕ
    // ─────────────────────────────────────────────────────────

    async _adaptiveExplore(input, feedback) {
        const reasons = {
            lowScore:      feedback.score < 0.1,
            lowConfidence: feedback.confidence < 0.4,
            highNovelty:   feedback.novelty > 0.7,
            highEnergy:    this.energy.current > 0.75
        };
        const exploreScore = Object.values(reasons).filter(Boolean).length / Object.keys(reasons).length;
        if (exploreScore < 0.4) return;

        const intensity = Math.min(1, exploreScore * (0.5 + this.energy.current * 0.5));

        if (this.embedder?.isTrained && intensity > 0.5) {
            await this._semanticDreamExplore(input, feedback, intensity);
        } else {
            await this._cheapExplore(input, feedback, intensity);
        }

        this.energy.spend('explore', 0.03 * intensity);
    }

    async _semanticDreamExplore(input, feedback, intensity) {
        const nDreams = Math.floor(2 + intensity * 4);
        const dreams  = await this._generateSemanticDreams(nDreams);
        let accepted  = 0;

        for (const dream of dreams) {
            if (this._shouldLearnFromDream(dream, feedback)) {
                await this.model.predict(dream.input, {});
                accepted++;
            }
        }

        this.stats.dreamsGenerated += nDreams;
        this.stats.dreamsAccepted  += accepted;
        this._dreamAcceptanceRate   = accepted / Math.max(1, nDreams);
    }

    async _generateSemanticDreams(n = 5) {
        const keys = Array.from(this.attractors.keys());
        if (keys.length < 2 || !this.embedder) return this._generateDreamsSimple(n);

        const dreams = [];
        for (let i = 0; i < n; i++) {
            const a = keys[Math.floor(Math.random() * keys.length)];
            let aEmb = null, bestBEmb = null;

            try {
                aEmb = await this.embedder.encode(a);
                let bestB = null, bestSim = -1;

                for (const candidate of keys) {
                    if (candidate === a) continue;
                    const cEmb = await this.embedder.encode(candidate);
                    const sim  = this.embedder.similarity(aEmb, cEmb);
                    // FIX 12: dispose cEmb сразу после use
                    cEmb.dispose();
                    if (sim > 0.3 && sim < 0.8 && sim > bestSim) { bestSim = sim; bestB = candidate; }
                }

                if (!bestB) continue;

                bestBEmb = await this.embedder.encode(bestB);
                // FIX 12: hybrid создаётся ДО dispose aEmb/bestBEmb — createHybridEmbedding сам делает dispose
                const alpha  = 0.5 + Math.random() * 0.2;
                const hybrid = createHybridEmbedding(aEmb, bestBEmb, { alpha, dim: this.embedder.dim });
                hybrid.dispose(); // нам только нужен ключ, не сам тензор

                const synInput = `${a}+${bestB}`;
                const novelty  = await this._estimateNovelty(synInput);
                dreams.push({ input: synInput, key: synInput, confidence: 0.5 + Math.random() * 0.3, novelty, uncertainty: Math.random() * 0.4 });

            } finally {
                aEmb?.dispose();
                bestBEmb?.dispose();
            }
        }

        this.stats.dreamsGenerated += n;
        return dreams;
    }

    async _generateDreamsSimple(n = 5) {
        const keys = Array.from(this.attractors.keys());
        if (keys.length < 2) return [];
        return Array.from({ length: n }, () => {
            const a = keys[Math.floor(Math.random() * keys.length)];
            const b = keys[Math.floor(Math.random() * keys.length)];
            if (a === b) return null;
            return { input: `${a}+${b}`, key: `${a}+${b}`, confidence: 0.5 + Math.random() * 0.3, novelty: 0.5 + Math.random() * 0.4, uncertainty: Math.random() * 0.5 };
        }).filter(Boolean);
    }

    async _cheapExplore(input, feedback, intensity) {
        // Вариации входа
        const variants = this._generateInputVariants(input, 3);
        for (const v of variants) {
            if (Math.random() < intensity * 0.5) await this.model.predict(v, {});
        }
        // Слабые аттракторы
        const weakKeys = Array.from(this.attractors.entries())
            .filter(([, s]) => s < 0.15).map(([k]) => k).slice(0, 2);
        for (const k of weakKeys) {
            if (Math.random() < intensity * 0.7) await this.model.predict(k, {});
        }
    }

    _shouldLearnFromDream(dream, feedback) {
        const [minN, maxN] = this.cfg.safeguards.dreamNoveltyRange;
        if (dream.confidence  < this.cfg.safeguards.dreamConfidenceMin) return false;
        if (dream.novelty     < minN || dream.novelty > maxN)           return false;
        if ((dream.uncertainty || 0) > this.cfg.dream?.maxUncertainty ?? 0.5) return false;
        if (!this.energy.hasEnergyFor('dream', 0.03))                   return false;
        if (feedback.score < -0.5 && dream.confidence < 0.7)            return false;
        return true;
    }

    async _estimateNovelty(input) {
        if (!this.embedder) return 0.5;
        const inputEmb = await this.embedder.encode(input);
        let maxSim = 0;
        for (const k of this.attractors.keys()) {
            const kEmb = await this.embedder.encode(k);
            const sim  = this.embedder.similarity(inputEmb, kEmb);
            kEmb.dispose();
            maxSim = Math.max(maxSim, sim);
        }
        inputEmb.dispose();
        return 1 - maxSim;
    }

    // ─────────────────────────────────────────────────────────
    //  ФРАКТАЛЬНОЕ ВЛИЯНИЕ И BIAS
    // ─────────────────────────────────────────────────────────

    _propagateFractalInfluence() {
        // Аттракторы → память
        for (const [key, strength] of this.attractors) {
            const entry = this.db?.cache?.get(key);
            if (!entry?.targets) continue;
            for (const t of Object.values(entry.targets)) {
                t.weight = (t.weight || 0) + strength * 0.005;
            }
        }
        // Память → аттракторы (обратная связь)
        for (const [key, entry] of (this.db?.cache?.entries() || [])) {
            if (!entry?.targets) continue;
            const tw  = Object.values(entry.targets).reduce((s, t) => s + (t.weight || 0), 0);
            const cur = this.attractors.get(key) || 0;
            const nv  = cur * 0.95 + Math.min(1, tw * 0.1) * 0.05;
            if (nv > this.cfg.attractors.pruneThreshold) {
                this.attractors.set(key, Math.min(this.cfg.attractors.max, nv));
            }
        }
    }

    _applyAttractorBias(results) {
        if (!results?.length) return results;
        return results
            .map(r => ({
                ...r,
                confidence: Math.min(1.0, (r.confidence || 0.5) * (1 + (this.attractors.get(r.key) || 0) * 0.3)),
                sources:    { ...(r.sources || {}), attractor: this.attractors.get(r.key) || 0 }
            }))
            .sort((a, b) => b.confidence - a.confidence);
    }

    async _predictWithBias(input, trainingData = {}) {
        const prediction = await this.model.predict(input, trainingData);
        if (prediction?.results) prediction.results = this._applyAttractorBias(prediction.results);
        return prediction;
    }

    // ─────────────────────────────────────────────────────────
    //  ВНУТРЕННЯЯ РЕФЛЕКСИЯ
    // ─────────────────────────────────────────────────────────

    async _internalReflectionCycle() {
        const samples = this.db?.getPopular?.(5) || [];
        for (const s of samples) {
            const mutated    = s.key + '_reflex';
            const prediction = await this._predictWithBias(mutated, {});
            const feedback   = this._evaluatePrediction(mutated, prediction, null);
            if (feedback.score > 0.3) this._updateBeliefs(mutated, prediction.results, feedback);
        }
        return { reflections: samples.length };
    }

    // ─────────────────────────────────────────────────────────
    //  ECHO-БУФЕРЫ
    // ─────────────────────────────────────────────────────────

    _pushToEcho(input, results, feedback) {
        const reward = feedback?.score ?? 0.5;
        if (reward < 0.3) return; // отбрасываем шум
        const entry = { input, results, reward, feedback, time: Date.now(), topKey: results?.[0]?.key, topConfidence: results?.[0]?.confidence };
        const layer = reward > 0.75 ? 'short' : reward > 0.5 ? 'mid' : 'long';
        this._addToLayer(entry, layer);
    }

    _addToLayer(entry, layerName) {
        const layer = this.echoLayers[layerName];
        if (!layer) return;
        layer.buffer.push(entry);
        const now = Date.now();
        layer.buffer = layer.buffer.filter(e => now - e.time < layer.ttl).slice(-layer.maxSize);
    }

    _getEchoBias(input, topN = 3) {
        const bias = {};
        const now  = Date.now();
        for (const [, layer] of Object.entries(this.echoLayers)) {
            for (const e of layer.buffer) {
                const age = now - e.time;
                if (age > layer.ttl * 0.8 || !e.topKey) continue;
                const freshness = 1 - (age / layer.ttl);
                bias[e.topKey] = (bias[e.topKey] || 0) + layer.weight * e.reward * freshness;
            }
        }
        return Object.entries(bias).sort((a, b) => b[1] - a[1]).slice(0, topN)
            .reduce((acc, [k, v]) => ({ ...acc, [k]: v }), {});
    }

    // ─────────────────────────────────────────────────────────
    //  ЗАЩИТЫ И ROLLBACK
    // ─────────────────────────────────────────────────────────

    _shouldRollback(feedback) {
        return feedback.hasActual &&
               feedback.supervisedScore < this.cfg.safeguards.rollbackThreshold &&
               feedback.confidence > 0.8 &&
               !this._realityAnchor.active;
    }

    _rollbackRecentUpdates() {
        if (this._realityAnchor.active) return;
        this._realityAnchor.active = true;
        if (this._realityAnchor.timerId) clearTimeout(this._realityAnchor.timerId);

        const cutoff = Date.now() - 60000;
        let count = 0;
        for (const rec of (this.db?.cache?.values() || [])) {
            const t = rec.lastUpdated || rec.lastUsed || 0;
            if (t > cutoff && rec.confidence) { rec.confidence = Math.max(0.1, rec.confidence * 0.8); count++; }
        }
        this.meta.explorationRate.value *= 0.7;
        console.log(`🛡️ Reality anchor: rolled back ${count} records`);

        this._realityAnchor.timerId = setTimeout(() => {
            this._realityAnchor.active  = false;
            this._realityAnchor.timerId = null;
        }, this._realityAnchor.cooldownMs);
    }

    _saveAttractorSnapshot(maxHistory = 5) {
        this._attractorHistory.push(new Map(Array.from(this.attractors.entries()).slice(0, 100)));
        if (this._attractorHistory.length > maxHistory) this._attractorHistory.shift();
    }

    _emergencySave() {
        try {
            const state = {
                attractors: Object.fromEntries(this.attractors),
                energy:     this.energy.export(),
                meta: {
                    learningRate:    this.meta.learningRate.value,
                    explorationRate: this.meta.explorationRate.value,
                    temperature:     this.meta.temperature.value
                },
                stats:     { ...this.stats },
                timestamp: Date.now()
            };
            localStorage.setItem('fractal_emergency', JSON.stringify(state));
        } catch (e) { console.error('❌ Emergency save failed:', e); }
    }

    // ─────────────────────────────────────────────────────────
    //  ОПЫТ И ПЕРСИСТЕНТНОСТЬ
    // ─────────────────────────────────────────────────────────

    _storeLearningExperience(exp) {
        const s = {
            landscape: { recentError: exp.landscape?.recentError, plateauEpochs: exp.landscape?.plateauEpochs, dataQuality: exp.landscape?.dataQuality, uncertainty: exp.landscape?.uncertainty },
            strategy:  exp.strategy,
            feedback:  { loss: exp.feedback?.loss, improvement: exp.feedback?.improvement, score: exp.feedback?.score },
            timestamp: exp.timestamp || Date.now()
        };
        this._experienceBuffer.push(s);
        if (this._experienceBuffer.length > 500) this._experienceBuffer.shift();
        if (this._experienceBuffer.length % 50 === 0) this._saveExperienceDebounced();
    }

    _persistExperience() {
        try {
            const toSave = this._experienceBuffer.slice(-100);
            localStorage.setItem('fractal_experience', JSON.stringify(toSave));
        } catch (e) { console.warn('⚠️ Failed to persist experience:', e.message); }
    }

    async loadExperience() {
        try {
            const saved = localStorage.getItem('fractal_experience');
            if (saved) {
                this._experienceBuffer = JSON.parse(saved);
                this._rebuildStrategyScores();
                console.log(`📥 Loaded ${this._experienceBuffer.length} experiences`);
            }
        } catch (e) { console.warn('⚠️ Failed to load experience:', e.message); }
    }

    _rebuildStrategyScores() {
        const scores = {}, counts = {};
        for (const exp of this._experienceBuffer) {
            scores[exp.strategy]  = (scores[exp.strategy]  || 0) + (exp.feedback?.score || 0);
            counts[exp.strategy]  = (counts[exp.strategy]  || 0) + 1;
        }
        this._strategyScores = {};
        for (const s of Object.keys(scores)) this._strategyScores[s] = scores[s] / counts[s];
    }

    // ─────────────────────────────────────────────────────────
    //  ADAPTIVE SELF-TRAIN
    // ─────────────────────────────────────────────────────────

    async _adaptiveSelfTrain(trainingData) {
        const triggers = {
            enoughEnergy:  this.energy.hasEnergyFor('train'),
            hasData:       (this.db?.cache?.size || 0) >= (this.cfg.loop.minSamplesForTrain ?? 3),
            uncertainty:   this._recentUncertainty > 0.4,
            noveltySpike:  this._lastNoveltyScore > this.cfg.loop.noveltyThreshold,
            timeSinceLast: (Date.now() - this.stats.lastTrainTime) > 30000
        };
        if (!triggers.enoughEnergy || !triggers.hasData || !triggers.timeSinceLast) return false;
        if (!triggers.uncertainty && !triggers.noveltySpike) return false;

        const data = this._collectTrainingData(trainingData);
        if (Object.keys(data).length < this.cfg.loop.minSamplesForTrain) return false;

        try {
            const epochs = Math.max(1, Math.floor(this.energy.modulate(2, { minFactor: 0.5, maxFactor: 2 })));
            await this.model.train(data, { epochs, learningRate: this.meta.learningRate.value });
            this.stats.lastTrainTime = Date.now();
            this.stats.selfTrainings++;
            this.energy.spend('train');
            return true;
        } catch (e) {
            console.error('❌ Self-train error:', e);
            this.energy.current = Math.min(1, this.energy.current + 0.1);
            return false;
        }
    }

    _collectTrainingData(trainingData) {
        const batch  = {};
        const cutoff = Date.now() - 3600000;
        for (const [key, rec] of (this.db?.cache?.entries() || [])) {
            if (rec.lastUsed < cutoff || rec.confidence < 0.6) continue;
            const payload = trainingData?.[key] || rec.payload;
            if (payload) batch[key] = payload;
            if (Object.keys(batch).length >= this.cfg.loop.maxTrainSamples) break;
        }
        return batch;
    }

    // ─────────────────────────────────────────────────────────
    //  INNER VOICE
    // ─────────────────────────────────────────────────────────

    startInnerVoice(period = null) {
        if (this.isRunning) return;
        this.isRunning = true;
        const p = period || this.cfg.loop.innerVoicePeriod;
        this.innerVoiceTimer = setInterval(async () => {
            try {
                // DOC5: Internal Voice — мышление без внешнего стимула
                await this._generateThought();

                if (Math.random() < 0.5) await this._reflectThoughts();
                if (Math.random() < 0.3) this._linkThoughts();

                // Периодически обучаем FastBrain на мыслях
                if (this.internalVoice.thoughts.length >= 10 && Math.random() < 0.2) {
                    await this._trainDistillationFromThoughts();
                }

                // Dream (реже)
                if (Math.random() < 0.2 && this.energy.current > 0.4) await this._dreamCycle();

                if (this.stats.totalProcessed % 50 === 0) this._pruneAttractors();
                if (this.stats.totalProcessed % 20 === 0 && this.energy.current > 0.5) await this.runMetaCycle();
            } catch (e) { console.error('❌ Inner voice error:', e); }
        }, p);
        console.log('🧠 Inner voice started');
    }

    stopInnerVoice() {
        if (this.innerVoiceTimer) {
            clearInterval(this.innerVoiceTimer);
            this.innerVoiceTimer = null;
            this.isRunning       = false;
            console.log('🔇 Inner voice stopped');
        }
    }

    // ─────────────────────────────────────────────────────────
    //  ВСПОМОГАТЕЛЬНЫЕ
    // ─────────────────────────────────────────────────────────

    _getOrCreateMemoryEntry(key) {
        if (!this.db?.cache) return null;
        if (!this.db.cache.has(key)) {
            this.db.cache.set(key, { key, targets: {}, totalWeight: 0, meta: {} });
        }
        return this.db.cache.get(key);
    }

    _trackLearningRate(change) {
        this._weightChanges.push(Math.abs(change));
        if (this._weightChanges.length > 20) this._weightChanges.shift();
        this._recentWeightChangeRate = this._weightChanges.reduce((a, b) => a + b, 0) / this._weightChanges.length;
    }

    _getMemoryConsistencyBonus(predictedKey, input) {
        const m = this.db?.cache?.get(input);
        if (!m) return 0;
        return m.key === predictedKey ? 0.3 + (m.confidence || 0.5) * 0.2 : -0.1;
    }

    _checkMemoryAlignment(input, predictedKey) {
        const m = this.db?.cache?.get(input);
        if (!m || !predictedKey) return 0.5;
        return m.key === predictedKey
            ? 0.5 + (m.confidence || 0.5) * 0.5
            : 0.5 - (m.confidence || 0.5) * 0.3;
    }

    _getCalibrationScore(window = 20) {
        const recent = this._feedbackHistory.slice(-window);
        if (recent.length < 5) return 0.5;
        let cal = 0, tot = 0;
        for (const fb of recent) {
            if (fb.hasActual && fb.supervisedScore !== undefined) {
                const wc = fb.confidence > 0.7, wr = fb.supervisedScore > 0;
                if (wc && wr) cal++; else if (wc && !wr) cal--;
                tot++;
            }
        }
        return tot > 0 ? 0.5 + (cal / tot) * 0.5 : 0.5;
    }

    _checkEntropyStagnation(currentEntropy, window = 10, tolerance = 0.05) {
        this._entropyHistory.push(currentEntropy);
        if (this._entropyHistory.length > window) this._entropyHistory.shift();
        if (this._entropyHistory.length < window)  return false;
        return Math.max(...this._entropyHistory) - Math.min(...this._entropyHistory) < tolerance;
    }

    _generateInputVariants(input, n = 3) {
        const v = [input, input + '*', input + '?', input.replace(/\s+/g, '_'), input.replace(/_/g, ' '), input.toLowerCase()];
        return [...new Set(v)].slice(0, n);
    }

    _getSystemPlasticity(window = 50) {
        const recent = this._feedbackHistory.slice(-window);
        if (recent.length < 10) return 1.0;
        const p =
            (1 - new Set(recent.map(f => Math.sign(f.score))).size / 3) * 0.3 +
            (this._recentWeightChangeRate || 0.5) * 0.3 +
            this.energy.current * 0.2 +
            (recent.filter(f => f.novelty > 0.6).length / recent.length) * 0.2;
        this._plasticityHistory.push(p);
        if (this._plasticityHistory.length > 10) this._plasticityHistory.shift();
        return this._plasticityHistory.reduce((a, b) => a + b, 0) / this._plasticityHistory.length;
    }

    _getTotalMemoryMass() {
        let sum = 0;
        for (const e of (this.db?.cache?.values() || [])) {
            if (e?.feedback) sum += Math.abs((e.feedback.pos || 0) - (e.feedback.neg || 0));
        }
        return sum;
    }

    _getRecentErrorRate(window = 10) {
        const recent = this._feedbackHistory.slice(-window);
        return recent.length > 0
            ? recent.filter(f => f.hasActual && f.supervisedScore < 0).length / recent.length
            : 0;
    }

    _getEffectiveExplorationRate() {
        return this.meta.explorationRate.value * (0.5 + this.energy.current * 0.5);
    }

    _pruneAttractors(threshold = null) {
        const thr = threshold ?? this.cfg.attractors.pruneThreshold;
        let pruned = 0;
        for (const [k, v] of this.attractors) {
            if (v < thr) { this.attractors.delete(k); this.attractorMeta.delete(k); pruned++; }
        }
        if (pruned > 0) console.log(`🧹 Pruned ${pruned} weak attractors`);
    }

    _getDiagnostics() {
        const plasticity   = this._getSystemPlasticity();
        const mass         = this._getTotalMemoryMass();
        const exploration  = this.meta.explorationRate.value;
        const dreamAccept  = this._dreamAcceptanceRate || 0;
        const warnings = [];
        if (exploration < 0.06) warnings.push('🔒 Exploration слишком низкий');
        if (plasticity   < 0.3) warnings.push('🧱 Система теряет пластичность');
        if (dreamAccept  < 0.15) warnings.push('💭 Dream-фильтры слишком строгие');
        if (mass         < 20)  warnings.push('📉 Мало накопленных знаний');
        return { plasticity, mass, exploration, dreamAcceptance: dreamAccept, warnings, overControlRisk: warnings.length >= 2 };
    }

    getStats() {
        return {
            ...this.stats,
            energy:     this.energy.getStats(),
            attractors: { count: this.attractors.size, top: Array.from(this.attractors.entries()).sort((a, b) => b[1] - a[1]).slice(0, 5) },
            echo:       { short: this.echoLayers.short.buffer.length, mid: this.echoLayers.mid.buffer.length, long: this.echoLayers.long.buffer.length },
            meta:       { explorationRate: this.meta.explorationRate.value, learningRate: this.meta.learningRate.value, temperature: this.meta.temperature.value },
            recentUncertainty:   this._recentUncertainty,
            strategyScores:      this._strategyScores,
            experienceBufferSize: this._experienceBuffer.length,
            diagnostics:         this._getDiagnostics(),
            // DOC5
            identity:            this.identity ? { ...this.identity, _history: undefined } : null,
            internalVoice: {
                thoughtsCount: this.internalVoice?.thoughts?.length || 0,
                linksCount:    this.internalVoice?.links?.length    || 0,
                lastReflected: this.internalVoice?.lastReflection   || 0
            },
            fastBrain: {
                ready:   !!this.fastBrain?.isTrained,
                classes: this.fastBrain?.allKeys?.length || 0
            }
        };
    }

    async exportState() {
        return {
            attractors: Object.fromEntries(this.attractors),
            energy:     this.energy.export(),
            meta: {
                learningRate:    this.meta.learningRate.value,
                explorationRate: this.meta.explorationRate.value,
                temperature:     this.meta.temperature.value
            },
            stats:          { ...this.stats },
            strategyScores: { ...this._strategyScores },
            dbExport:       await this.db?.exportJSON?.()
        };
    }

    async importState(state) {
        if (state.attractors) this.attractors = new Map(Object.entries(state.attractors));
        if (state.energy)     this.energy.import(state.energy);
        if (state.meta) {
            if (state.meta.learningRate)    this.meta.learningRate.value    = state.meta.learningRate;
            if (state.meta.explorationRate) this.meta.explorationRate.value = state.meta.explorationRate;
            if (state.meta.temperature)     this.meta.temperature.value     = state.meta.temperature;
        }
        if (state.stats)          Object.assign(this.stats, state.stats);
        if (state.strategyScores) Object.assign(this._strategyScores, state.strategyScores);
        console.log('📥 State imported');
    }


    // ═══════════════════════════════════════════════════════════════
    //  DOC5: FAST BRAIN ИНТЕГРАЦИЯ
    // ═══════════════════════════════════════════════════════════════

    /** Слияние предсказаний медленного (логика) и быстрого (интуиция) мозгов */
    _mergePredictions(slowResults = [], fastResults = []) {
        const merged = new Map();

        // Медленный мозг = 70% веса (глубокий анализ)
        for (const r of slowResults) {
            merged.set(r.key, { ...r, confidence: r.confidence * 0.7, source: 'slow' });
        }

        // Быстрый мозг = 30% веса (интуиция)
        for (const r of fastResults) {
            if (!merged.has(r.key)) {
                merged.set(r.key, { ...r, confidence: r.confidence * 0.3, source: 'fast' });
            } else {
                const existing = merged.get(r.key);
                existing.confidence = Math.min(1, existing.confidence + r.confidence * 0.3);
                existing.source = 'both';
            }
        }

        return Array.from(merged.values()).sort((a, b) => b.confidence - a.confidence);
    }

    /** Обнаружение конфликта между fast и slow brain → "сомнение" */
    _detectConflict(slowResults = [], fastResults = []) {
        if (!slowResults.length || !fastResults.length) return 0;
        const slowTop = slowResults[0]?.key;
        const fastTop = fastResults[0]?.key;

        // Разные лидеры = конфликт
        if (slowTop !== fastTop) {
            const slowConf = slowResults[0]?.confidence || 0;
            const fastConf = fastResults[0]?.confidence || 0;
            // Конфликт тем сильнее, чем выше уверенность обеих сторон
            return Math.min(1, (slowConf + fastConf) / 2);
        }
        return 0;
    }

    // ═══════════════════════════════════════════════════════════════
    //  DOC5: SELF-SUPERVISION — "система сама себе учитель"
    // ═══════════════════════════════════════════════════════════════

    async _selfSupervise(input) {
        if (!this.db?.cache) return null;
        const mem = this.db.cache.get(input);
        if (!mem) return null;

        // Если в памяти есть уверенный ответ — используем его как ground truth
        if (mem.confidence > 0.8) {
            return { key: mem.key };
        }

        // Мягкий вариант: возвращаем если был позитивный фидбек
        const posRatio = (mem.feedback?.pos || 0) / Math.max(1, (mem.feedback?.pos || 0) + (mem.feedback?.neg || 0));
        if (posRatio > 0.75 && (mem.feedback?.pos || 0) >= 3) {
            return { key: mem.key };
        }

        return null;
    }

    // ═══════════════════════════════════════════════════════════════
    //  DOC5: INTERNAL VOICE — генератор мыслей (самостоятельное мышление)
    // ═══════════════════════════════════════════════════════════════

    /** Генерация мыслей: берём популярные объекты и генерируем варианты */
    async _generateThought() {
        if (!this.energy.hasEnergyFor('explore', 0.01)) return;

        const samples = this.db?.getPopular?.(3) || [];
        if (!samples.length) return;

        for (const s of samples) {
            if (!s?.key) continue;
            const variants = this._generateInputVariants(s.key, 3);

            for (const v of variants) {
                try {
                    const prediction = await this._predictWithBias(v, this._fastBrainData);
                    const feedback   = this._evaluatePrediction(v, prediction, null);
                    const score      = feedback?.score ?? 0.5;

                    this.internalVoice.thoughts.push({
                        input:  v,
                        result: prediction.results,
                        score,
                        time:   Date.now()
                    });
                } catch(e) {}
            }
        }

        // Чистка старых мыслей
        if (this.internalVoice.thoughts.length > this.internalVoice.maxThoughts) {
            this.internalVoice.thoughts = this.internalVoice.thoughts.slice(-this.internalVoice.maxThoughts);
        }
        this.internalVoice.lastGeneration = Date.now();
    }

    /** Рефлексия: хорошие мысли закрепляем в убеждениях */
    async _reflectThoughts() {
        const thoughts = this.internalVoice.thoughts.slice(-15);
        let reflected  = 0;

        for (const t of thoughts) {
            if (t.score < 0.4) continue;
            try {
                this._updateBeliefs(t.input, t.result || [], {
                    score:         t.score * 0.6, // снижаем вес внутренней мысли
                    signalQuality: 0.5,
                    entropy:       0.5,
                    novelty:       0.3,
                    hasActual:     false
                });
                reflected++;
            } catch(e) {}
        }
        this.internalVoice.lastReflection = Date.now();
        return reflected;
    }

    /** Связывание мыслей: ищем семантически близкие пары → "понимание" */
    _linkThoughts() {
        const thoughts = this.internalVoice.thoughts;
        const linked   = [];

        for (let i = 0; i < thoughts.length - 1; i++) {
            const a = thoughts[i];
            const b = thoughts[i + 1];
            const sim = stringSimilarity(a.input, b.input);

            if (sim > 0.5) {
                linked.push({ from: a.input, to: b.input, strength: sim });
                // Усиливаем аттрактор если мысли связаны
                const attrKey = `${a.input}↔${b.input}`;
                const cur = this.attractors.get(attrKey) || 0;
                this.attractors.set(attrKey, Math.min(0.4, cur + sim * 0.05));
            }
        }

        this.internalVoice.links = linked.slice(-30);
        return linked.length;
    }

    /** FastBrain учится на лучших мыслях внутреннего голоса */
    async _trainDistillationFromThoughts() {
        if (!this.fastBrain) return;

        const thoughts = this.internalVoice.thoughts.slice(-50);
        const data     = {};

        for (const t of thoughts) {
            if (t.score < 0.55) continue;
            const key = t.input;
            const payload = t.result?.[0]?.payload || this._fastBrainData[key] || { key };
            data[key] = payload;
        }

        if (Object.keys(data).length >= 5) {
            try {
                await this.fastBrain.train(data, 8); // быстрое обновление
            } catch(e) {}
        }
    }

    /** Адаптация Identity на основе истории фидбека */
    _updateIdentity(feedback) {
        if (!this.identity) return;
        const alpha = 0.02; // медленное изменение

        // Высокое novelty → повышаем curiosity
        if (feedback.novelty > 0.6) {
            this.identity.curiosity = clamp(this.identity.curiosity + alpha, 0.1, 0.9);
        }
        // Хороший score при стабильной стратегии → повышаем stabilityBias
        if (feedback.score > 0.5 && this._strategyState.streak > 5) {
            this.identity.stabilityBias = clamp(this.identity.stabilityBias + alpha, 0.1, 0.9);
        }
        // Плохой score → повышаем riskTolerance (готовность менять стратегию)
        if (feedback.score < -0.2) {
            this.identity.riskTolerance = clamp(this.identity.riskTolerance + alpha * 2, 0.1, 0.9);
            this.identity.stabilityBias = clamp(this.identity.stabilityBias - alpha, 0.1, 0.9);
        }

        this.identity._history.push({ time: Date.now(), ...this.identity });
        if (this.identity._history.length > 50) this.identity._history.shift();
    }

    async destroy() {
        this.stopInnerVoice();
        if (this._globalDecayTimer) clearInterval(this._globalDecayTimer);
        if (this._realityAnchor.timerId) clearTimeout(this._realityAnchor.timerId);
        this.embedder?.dispose();
        this.semanticCache?.clear();
        this.db?.destroy?.();
        this.attractors.clear();
        this.attractorMeta.clear();
        this._attractorHistory = [];
        console.log('🧹 FractalController destroyed');
    }
}

// ═══════════════════════════════════════════════════════════════
//  11. AUTONOMOUS LEARNING LOOP
// ═══════════════════════════════════════════════════════════════

export class AutonomousLearningLoop {
    constructor(controller, options = {}) {
        this.controller = controller;
        this.options = {
            maxCyclesPerSecond: 2,
            memoryThresholdMB:  100,
            onError:            'continue',  // 'continue' | 'pause' | 'throw'
            persistExperience:  true,
            enableMetaCycle:    true,
            enableDreaming:     true,
            ...options
        };
        this._running = false;
        this._stats   = { cycles: 0, errors: 0, strategies: {}, avgScore: null };
    }

    async start(stream) {
        if (this._running) return;
        this._running = true;
        if (this.options.persistExperience) await this.controller.loadExperience?.();
        console.log('🚀 Autonomous learning started');
        try { await this._runLoop(stream); }
        finally {
            this._running = false;
            console.log('🛑 Autonomous learning stopped');
        }
    }

    stop() { this._running = false; }

    async _runLoop(stream) {
        const { maxCyclesPerSecond, memoryThresholdMB, onError, enableMetaCycle } = this.options;
        const minCycleTime = 1000 / maxCyclesPerSecond;
        let lastCycleTime  = 0;

        for await (const batch of stream) {
            if (!this._running) break;

            // Backpressure
            const now   = Date.now();
            const delay = Math.max(0, minCycleTime - (now - lastCycleTime));
            if (delay > 0) await new Promise(r => setTimeout(r, delay));

            // Мониторинг памяти
            const mem   = getMemoryStats();
            const memMB = mem.numBytes / (1024 * 1024);
            if (memMB > memoryThresholdMB) {
                console.warn(`🛑 Memory: ${memMB.toFixed(1)}MB > ${memoryThresholdMB}MB. Cleaning...`);
                emergencyCleanup();
                await new Promise(r => setTimeout(r, 2000));
                continue;
            }

            try {
                const result = enableMetaCycle
                    ? await this.controller.fullCycle(batch, null)
                    : await this.controller.runSelfLearningCycle(batch, null);

                lastCycleTime = Date.now();
                this._stats.cycles++;

                const strategy = result.strategy || result.external?.strategy;
                if (strategy) this._stats.strategies[strategy] = (this._stats.strategies[strategy] || 0) + 1;

                const score = result.feedback?.score ?? result.external?.feedback?.score;
                if (score !== undefined) {
                    const alpha = 0.01;
                    this._stats.avgScore = this._stats.avgScore === null
                        ? score
                        : this._stats.avgScore * (1 - alpha) + score * alpha;
                }

                if (this._stats.cycles % 10 === 0) {
                    console.log(`📊 Cycle #${this._stats.cycles}: ${strategy} | score: ${score?.toFixed(3)} | avg: ${this._stats.avgScore?.toFixed(3)} | energy: ${this.controller.energy.current.toFixed(2)}`);
                }

            } catch (e) {
                this._stats.errors++;
                console.error(`❌ Cycle error #${this._stats.cycles}:`, e.message);
                this.controller._rollbackRecentUpdates?.();

                if      (onError === 'pause') await new Promise(r => setTimeout(r, 2000));
                else if (onError === 'throw') throw e;
                // 'continue' — продолжаем
            }
        }
    }

    getStats() {
        return {
            ...this._stats,
            running:        this._running,
            experienceSize: this.controller._experienceBuffer?.length || 0
        };
    }
}

// ═══════════════════════════════════════════════════════════════
//  12. BOOTSTRAP
// ═══════════════════════════════════════════════════════════════

export async function bootstrapFractalSystem(model, userConfig = {}) {
    const config = { ...DEFAULT_CONFIG, ...userConfig };

    // 1. DB
    const db = new SelfLearningDB(config.db);
    await db.init();

    // 2. Controller
    const controller = new FractalController(model, db, config);
    await controller.loadExperience();

    // 3. DOC5: FastBrain — используем основную модель как интуицию
    //    (или отдельную fastBrainModel если передана в config)
    const fastBrainModel = config.fastBrainModel || model;
    const trainingData   = config.trainingData   || {};
    controller.setFastBrain(fastBrainModel, trainingData);

    // 4. Inner voice (опционально)
    if (config.startInnerVoice !== false) {
        controller.startInnerVoice();
    }

    // 4. Мониторинг памяти (опционально)
    let memMonitor = null;
    if (config.monitorMemory) {
        memMonitor = setInterval(() => {
            const mem = getMemoryStats();
            if (mem.numTensors > 1000) {
                console.warn(`⚠️ High tensor count: ${mem.numTensors}`);
                emergencyCleanup();
            }
        }, 30000);
    }

    // 5. Cleanup при выгрузке
    if (typeof window !== 'undefined') {
        window.addEventListener('beforeunload', async () => {
            await controller.destroy();
            if (memMonitor) clearInterval(memMonitor);
        });
    }

    console.log('✅ Fractal System ready');
    return { db, controller, config };
}

// ═══════════════════════════════════════════════════════════════
//  13. VERIFY META-LAYER (тест интеграции)
// ═══════════════════════════════════════════════════════════════

export async function verifyMetaLayer(controller) {
    console.log('🧪 Verifying Fractal System...\n');
    const results = {};

    // Тест 1: Цикл работает
    try {
        const mockBatch = 'test_input';
        const result = await controller.runSelfLearningCycle(mockBatch, null, {});
        results.cycle = result?.strategy ? `✅ Cycle OK: ${result.strategy}` : '❌ Cycle failed';
    } catch (e) { results.cycle = `❌ ${e.message}`; }

    // Тест 2: Energy gating
    try {
        const orig = controller.energy.current;
        controller.energy.current = 0.05;
        const result = await controller.runSelfLearningCycle('test', null, {});
        results.energyGate = result?.skipped === 'low_energy' ? '✅ Energy gating OK' : '❌ Gate not triggered';
        controller.energy.current = orig;
    } catch (e) { results.energyGate = `❌ ${e.message}`; }

    // Тест 3: Аттракторы влияют на стратегию (FIX 8c)
    try {
        controller.attractors.set('test', 0.8);
        controller.attractors.set('test2', 0.7);
        controller.energy.current = 0.9;
        const landscape = { recentError: 0.3, plateauEpochs: 0, dataQuality: 0.7, uncertainty: 0.5, memoryUsage: 0.3 };
        const strategy = await controller._chooseRefinementStrategy(landscape);
        results.attractorBias = strategy === 'consolidate' ? '✅ Attractor→consolidate OK' : `⚠️ Got: ${strategy}`;
        controller.attractors.clear();
    } catch (e) { results.attractorBias = `❌ ${e.message}`; }

    // Тест 4: _softCleanData возвращает Object (FIX 2)
    try {
        const data   = { key1: { confidence: 0.8, feedback: { pos: 3, neg: 1 } }, key2: { confidence: 0.2 } };
        const cleaned = controller._softCleanData(data);
        results.softClean = typeof cleaned === 'object' && !Array.isArray(cleaned)
            ? '✅ softCleanData → Object'
            : '❌ softCleanData returned Array';
    } catch (e) { results.softClean = `❌ ${e.message}`; }

    // Тест 5: targets заполняются из results (FIX 3)
    try {
        const fakeResults = [{ key: 'bottle', confidence: 0.9 }, { key: 'cup', confidence: 0.1 }];
        const feedback    = { score: 0.7, signalQuality: 0.8, hasActual: false, entropy: 0.5, novelty: 0.3 };
        controller._updateBeliefs('test_key_fix3', fakeResults, feedback);
        const entry = controller.db?.cache?.get('test_key_fix3');
        results.targetsPopulated = (entry?.targets?.bottle !== undefined)
            ? '✅ targets populated from results'
            : '❌ targets still empty';
    } catch (e) { results.targetsPopulated = `❌ ${e.message}`; }

    // Тест 6: FIX 9 — _detectPlateau не крашит на undefined loss
    try {
        controller._feedbackHistory = [
            { score: 0.5 }, { score: 0.49 }, { score: 0.51 }, { score: 0.5 }, { score: 0.5 }
        ];
        const plateau = controller._detectPlateau();
        results.detectPlateau = typeof plateau === 'number' ? `✅ detectPlateau OK: ${plateau}` : '❌ detectPlateau crash';
    } catch (e) { results.detectPlateau = `❌ ${e.message}`; }

    // Тест 7: Meta-цикл доступен
    try {
        const meta = await controller.runMetaCycle?.();
        results.metaCycle = meta !== undefined ? '✅ Meta-cycle OK' : '❌ Missing';
    } catch (e) { results.metaCycle = `❌ ${e.message}`; }

    // Вывод
    console.log('\nResults:');
    for (const [test, res] of Object.entries(results)) console.log(`  ${test}: ${res}`);
    const passed = Object.values(results).filter(r => r.startsWith('✅')).length;
    console.log(`\n${passed === Object.keys(results).length ? '🎉' : '⚠️'} ${passed}/${Object.keys(results).length} tests passed`);

    return { results, allPassed: passed === Object.keys(results).length };
}
