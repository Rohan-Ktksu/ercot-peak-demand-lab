import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
const root = path.resolve('docs');
const types = {'.html':'text/html','.css':'text/css','.mjs':'text/javascript','.json':'application/json','.zip':'application/zip','.ipynb':'application/x-ipynb+json','.py':'text/plain'};
const server = http.createServer((request,response) => {
  const target = path.resolve(root,'.'+decodeURIComponent(new URL(request.url,'http://localhost').pathname));
  if (target !== root && !target.startsWith(root + path.sep)) {response.writeHead(403).end();return;}
  const filename = target === root ? path.join(root,'index.html') : target;
  if (!fs.existsSync(filename) || !fs.statSync(filename).isFile()) {response.writeHead(404).end('Not found');return;}
  response.setHeader('Content-Type',types[path.extname(filename)] ?? 'application/octet-stream');
  fs.createReadStream(filename).pipe(response);
});
server.listen(4173,'127.0.0.1',()=>console.log('Local: http://127.0.0.1:4173'));
