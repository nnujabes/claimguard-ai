/* Keep analysis images in IndexedDB rather than the small localStorage quota. */
window.claimStore = (() => {
  let database;
  function open() {
    if (!database) database = new Promise((resolve, reject) => {
      const request = indexedDB.open('claimguard-claims', 1);
      request.onupgradeneeded = () => {
        request.result.createObjectStore('claims', {keyPath: 'id'});
        request.result.createObjectStore('meta');
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    return database;
  }
  async function transaction(stores, mode, action) {
    const db = await open();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(stores, mode);
      let result;
      tx.oncomplete = () => resolve(result);
      tx.onabort = () => reject(tx.error || new Error('저장 작업이 중단되었습니다.'));
      tx.onerror = () => reject(tx.error);
      action(tx, value => { result = value; });
    });
  }
  async function list() {
    // Preserve old score-only claims, but never resurrect a deleted claim.
    let legacy = [];
    try { legacy = JSON.parse(localStorage.getItem('cg-real-claims-v1') || '[]'); } catch (_) {}
    await transaction(['claims', 'meta'], 'readwrite', tx => {
      const migrated = tx.objectStore('meta').get('migrated-v1');
      migrated.onsuccess = () => {
        if (migrated.result) return;
        if (Array.isArray(legacy)) legacy.forEach(claim => {
          if (claim && typeof claim.id === 'string' && Array.isArray(claim.images)) tx.objectStore('claims').put(claim);
        });
        tx.objectStore('meta').put(true, 'migrated-v1');
      };
    });
    return transaction(['claims'], 'readonly', (tx, done) => {
      const request = tx.objectStore('claims').getAll();
      request.onsuccess = () => done(request.result);
    });
  }
  const put = claim => transaction(['claims'], 'readwrite', tx => tx.objectStore('claims').put(claim));
  const remove = id => transaction(['claims'], 'readwrite', tx => tx.objectStore('claims').delete(id));
  const getMeta = key => transaction(['meta'], 'readonly', (tx, done) => {
    const request = tx.objectStore('meta').get(key);
    request.onsuccess = () => done(request.result);
  });
  const setMeta = (key, value) => transaction(['meta'], 'readwrite', tx => tx.objectStore('meta').put(value, key));
  return {list, put, remove, getMeta, setMeta};
})();
