import asyncio
import websockets
from aiohttp import web
from .. import compute_face_detection

# --- WebSocket Handler ---
async def websocket_handler(websocket, path):
    """
    Asynchronous handler function for WebSocket connections.
    It receives messages and echoes them back to the client.
    """
    print(f"WebSocket client connected from path: {path}")
    try:
        async for message in websocket:
            print(f"Received from client: {message}")
            await websocket.send(f"Server echo: {message}")
    except websockets.exceptions.ConnectionClosedOK:
        print("WebSocket client disconnected gracefully.")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"WebSocket client disconnected with error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in WebSocket: {e}")
    finally:
        print("WebSocket connection closed.")

# --- HTTP Handler ---
async def http_handler(request):
    """
    Asynchronous handler function for HTTP GET requests.
    Serves a simple HTML page that connects to the WebSocket server.
    """
    

    return web.Response(text="<html><h1>Kabeer </h1></html>", content_type='text/html')

async def main():
    """
    Main function to start both HTTP and WebSocket servers concurrently.
    """
    # --- Start WebSocket Server ---
    websocket_server = websockets.serve(websocket_handler, "localhost", 8765)
    print("WebSocket server started on ws://localhost:8765")

    # --- Start HTTP Server ---
    app = web.Application()
    app.router.add_get('/', http_handler) # Serve the HTML page at the root URL
    runner = web.AppRunner(app)
    await runner.setup()
    http_site = web.TCPSite(runner, 'localhost', 8080) # HTTP server on port 8080
    await http_site.start()
    print("HTTP server started on http://localhost:8080")

    # Keep both servers running indefinitely
    await asyncio.Future() # This will run forever until cancelled

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServers stopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")

