const redis = require('redis');
const crypto = require('crypto');
const logger = require('./logger');

// Redis configuration
const REDIS_ENABLED = process.env.REDIS_ENABLED !== 'false';
const REDIS_URL = process.env.REDIS_URL || 'redis://localhost:6379';
const CACHE_TTL = parseInt(process.env.CACHE_TTL || '86400', 10); // 24 hours default
const CACHE_PREFIX = 'zkx402:proof:';

let redisClient = null;
let cacheStats = {
  hits: 0,
  misses: 0,
  errors: 0,
};

// Initialize Redis client
const initializeRedis = async () => {
  if (!REDIS_ENABLED) {
    logger.info('Redis caching is disabled');
    return null;
  }

  try {
    redisClient = redis.createClient({
      url: REDIS_URL,
      socket: {
        connectTimeout: 5000,
        reconnectStrategy: (retries) => {
          if (retries > 10) {
            logger.error('Redis reconnection failed after 10 attempts');
            return new Error('Redis reconnection limit exceeded');
          }
          const delay = Math.min(retries * 100, 3000);
          logger.warn(`Redis reconnecting in ${delay}ms (attempt ${retries})`);
          return delay;
        },
      },
    });

    redisClient.on('error', (err) => {
      logger.error('Redis client error', { error: err.message });
      cacheStats.errors++;
    });

    redisClient.on('connect', () => {
      logger.info('Redis client connected');
    });

    redisClient.on('ready', () => {
      logger.info('Redis client ready');
    });

    redisClient.on('reconnecting', () => {
      logger.warn('Redis client reconnecting');
    });

    await redisClient.connect();
    logger.info('Redis cache initialized successfully', { url: REDIS_URL });
    return redisClient;
  } catch (error) {
    logger.error('Failed to initialize Redis cache', { error: error.message });
    redisClient = null;
    return null;
  }
};

// Generate cache key from proof parameters
const generateCacheKey = (modelId, inputs) => {
  // Create a deterministic hash of the model and inputs
  const inputsStr = Array.isArray(inputs) ? inputs.join(',') : JSON.stringify(inputs);
  const hash = crypto
    .createHash('sha256')
    .update(`${modelId}:${inputsStr}`)
    .digest('hex');
  return `${CACHE_PREFIX}${hash}`;
};

// Get proof from cache
const getProofFromCache = async (modelId, inputs) => {
  if (!REDIS_ENABLED || !redisClient || !redisClient.isOpen) {
    return null;
  }

  try {
    const key = generateCacheKey(modelId, inputs);
    const cachedData = await redisClient.get(key);

    if (cachedData) {
      cacheStats.hits++;
      logger.info('Cache hit', {
        modelId,
        cacheKey: key,
        stats: { ...cacheStats },
      });
      return JSON.parse(cachedData);
    }

    cacheStats.misses++;
    logger.debug('Cache miss', {
      modelId,
      cacheKey: key,
      stats: { ...cacheStats },
    });
    return null;
  } catch (error) {
    cacheStats.errors++;
    logger.error('Error getting proof from cache', {
      modelId,
      error: error.message,
    });
    return null;
  }
};

// Store proof in cache
const storeProofInCache = async (modelId, inputs, proofResult) => {
  if (!REDIS_ENABLED || !redisClient || !redisClient.isOpen) {
    return false;
  }

  try {
    const key = generateCacheKey(modelId, inputs);
    const cacheData = {
      ...proofResult,
      cachedAt: new Date().toISOString(),
      modelId,
    };

    await redisClient.setEx(key, CACHE_TTL, JSON.stringify(cacheData));

    logger.info('Proof stored in cache', {
      modelId,
      cacheKey: key,
      ttl: CACHE_TTL,
    });
    return true;
  } catch (error) {
    cacheStats.errors++;
    logger.error('Error storing proof in cache', {
      modelId,
      error: error.message,
    });
    return false;
  }
};

// Clear cache for a specific model
const clearModelCache = async (modelId) => {
  if (!REDIS_ENABLED || !redisClient || !redisClient.isOpen) {
    return 0;
  }

  try {
    const pattern = `${CACHE_PREFIX}*`;
    let cursor = 0;
    let deletedCount = 0;

    do {
      const reply = await redisClient.scan(cursor, {
        MATCH: pattern,
        COUNT: 100,
      });

      cursor = reply.cursor;
      const keys = reply.keys;

      if (keys.length > 0) {
        for (const key of keys) {
          // Get the cached data to check if it matches the modelId
          const data = await redisClient.get(key);
          if (data) {
            const parsed = JSON.parse(data);
            if (parsed.modelId === modelId) {
              await redisClient.del(key);
              deletedCount++;
            }
          }
        }
      }
    } while (cursor !== 0);

    logger.info('Cleared model cache', { modelId, deletedCount });
    return deletedCount;
  } catch (error) {
    logger.error('Error clearing model cache', {
      modelId,
      error: error.message,
    });
    return 0;
  }
};

// Clear all cache
const clearAllCache = async () => {
  if (!REDIS_ENABLED || !redisClient || !redisClient.isOpen) {
    return 0;
  }

  try {
    const pattern = `${CACHE_PREFIX}*`;
    let cursor = 0;
    let deletedCount = 0;

    do {
      const reply = await redisClient.scan(cursor, {
        MATCH: pattern,
        COUNT: 100,
      });

      cursor = reply.cursor;
      const keys = reply.keys;

      if (keys.length > 0) {
        await redisClient.del(keys);
        deletedCount += keys.length;
      }
    } while (cursor !== 0);

    logger.info('Cleared all cache', { deletedCount });
    return deletedCount;
  } catch (error) {
    logger.error('Error clearing all cache', { error: error.message });
    return 0;
  }
};

// Get cache statistics
const getCacheStats = () => {
  const hitRate =
    cacheStats.hits + cacheStats.misses > 0
      ? ((cacheStats.hits / (cacheStats.hits + cacheStats.misses)) * 100).toFixed(2)
      : 0;

  return {
    ...cacheStats,
    hitRate: `${hitRate}%`,
    enabled: REDIS_ENABLED,
    connected: redisClient?.isOpen || false,
  };
};

// Graceful shutdown
const closeRedis = async () => {
  if (redisClient && redisClient.isOpen) {
    try {
      await redisClient.quit();
      logger.info('Redis client closed gracefully');
    } catch (error) {
      logger.error('Error closing Redis client', { error: error.message });
    }
  }
};

module.exports = {
  initializeRedis,
  getProofFromCache,
  storeProofInCache,
  clearModelCache,
  clearAllCache,
  getCacheStats,
  closeRedis,
};
