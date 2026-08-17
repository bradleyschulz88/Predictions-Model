// Replaced at build time by scripts/stamp_service_worker.py with a hash of the
// files in ASSETS below. `activate` deletes every cache that is not the current
// one, so this name is what evicts a previous build's index.html/app.js/
// styles.css from a browser that already has them.
//
// It used to be bumped by hand, and the comment saying so was the whole
// mechanism. It sat at v53 from 2026-07-30 while 23 commits changed board.js
// and app.js, so the eviction path did not run for three weeks. Network-first
// fetching hid it while the network answered; a Pages outage is when it would
// not have. The value below is only the local-development fallback -- what
// ships is derived.
const CACHE = "edge-board-dev";
const ASSETS = [
  "./",
  "./index.html",
  "./board.css",
  "./board.js",
  // The legacy bet tracker and its stylesheet still ship, at tools.html.
  "./tools.html",
  "./app.js",
  "./styles.css",
  "./manifest.json",
];

self.addEventListener("install", (event) => {
  self.skipWaiting();
  event.waitUntil(caches.open(CACHE).then((cache) => cache.addAll(ASSETS)));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((key) => key !== CACHE).map((key) => caches.delete(key))))
      .then(() => self.clients.claim())
  );
});

function isDataJson(url) {
  return url.pathname.includes("/data/") && url.pathname.endsWith(".json");
}

function isAppShell(url) {
  return (
    url.pathname.endsWith(".html") ||
    url.pathname.endsWith(".js") ||
    url.pathname.endsWith(".css") ||
    url.pathname.endsWith("/")
  );
}

self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") {
    return;
  }

  const url = new URL(event.request.url);

  if (isDataJson(url) || isAppShell(url)) {
    // Network-first is only actually first if it reaches the network. A plain
    // fetch() still goes through the browser's HTTP cache, so a shell asset
    // GitHub Pages served with max-age could be replayed from disk and a new
    // deploy would keep showing the old dashboard. Revalidate instead.
    event.respondWith(
      fetch(event.request, { cache: "no-cache" })
        .then((response) => {
          if (response.ok) {
            const clone = response.clone();
            caches.open(CACHE).then((cache) => cache.put(event.request, clone));
          }
          return response;
        })
        .catch(() => caches.match(event.request))
    );
    return;
  }

  event.respondWith(
    caches.match(event.request).then((cached) => {
      const network = fetch(event.request)
        .then((response) => {
          if (response.ok) {
            caches.open(CACHE).then((cache) => cache.put(event.request, response.clone()));
          }
          return response;
        })
        .catch(() => cached);
      return cached || network;
    })
  );
});
