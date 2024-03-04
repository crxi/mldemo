import argparse
import uvicorn
import streamsync.serve
from fastapi import FastAPI, Response
from contextlib import asynccontextmanager

parser = argparse.ArgumentParser(description='serve.py')
parser.add_argument('--loglevel', type=str, default='info',
                    help='Log level for the server script')
args = parser.parse_args()

knn_app = streamsync.serve.get_asgi_app("./knn", "run")
wwt_app = streamsync.serve.get_asgi_app("./wwt", "run")
kcl_app = streamsync.serve.get_asgi_app("./kcl", "run")

@asynccontextmanager
async def lifespan_context(app: FastAPI):
    async with knn_app.router.lifespan_context(app):
        async with wwt_app.router.lifespan_context(app):
            async with kcl_app.router.lifespan_context(app):
                yield

root_asgi_app = FastAPI(lifespan=lifespan_context)
root_asgi_app.mount("/knn", knn_app)
root_asgi_app.mount("/wwt", wwt_app)
root_asgi_app.mount("/kcl", kcl_app)

@root_asgi_app.get("/")
async def init():
    with open('static/index.htm') as fd:
        return Response(fd.read())

uvicorn.run(root_asgi_app,
    host="0.0.0.0",
    port=3333,
    log_level=args.loglevel,
    ws_max_size=streamsync.serve.MAX_WEBSOCKET_MESSAGE_SIZE)
