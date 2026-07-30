// Bump on any app-shell change. `activate` deletes every cache that is not this
// one, so a bump is what evicts a previous build's index.html/app.js/styles.css
// from a browser that already has them.
const CACHE = "edge-board-v53";
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
