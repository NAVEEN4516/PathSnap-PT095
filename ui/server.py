"""Laptop 2. Reverse proxy only the two required API operations."""
import json,os,re
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.error import HTTPError,URLError
from urllib.request import Request,urlopen
from urllib.parse import urlsplit

BASE=os.environ.get("PATHSNAP_API","http://127.0.0.1:8000").rstrip("/")
class UI(BaseHTTPRequestHandler):
    def reply(self,status,body,mime):
        self.send_response(status)
        self.send_header("Content-Type",mime)
        self.send_header("Content-Length",str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def proxy(self,path):
        try:
            req=Request(BASE+path,method=self.command)
            with urlopen(req,timeout=180) as remote:
                self.reply(remote.status,remote.read(),"application/json")
        except HTTPError as exc:
            self.reply(exc.code,exc.read(),"application/json")
        except (URLError,TimeoutError):
            self.reply(502,b'{"detail":"Laptop 1 could not be reached. Check its API address and network."}',"application/json")
    def do_GET(self):
        path=urlsplit(self.path).path
        if path=="/":
            self.reply(200,Path(__file__).with_name("index.html").read_bytes(),"text/html; charset=utf-8")
        elif path=="/trips":self.proxy(path)
        else:self.send_error(404)
    def do_POST(self):
        if re.fullmatch(r"/match/[0-9]{8}",self.path):self.proxy(self.path)
        else:self.send_error(404)

if __name__=="__main__":
    print("Open http://127.0.0.1:8501 on Laptop 2",flush=True)
    ThreadingHTTPServer(("127.0.0.1",8501),UI).serve_forever()
