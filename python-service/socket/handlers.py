from flask import request

def register_socket_handlers(socketio):
    @socketio.on('connect')
    def handle_connect():
        print('Client connected:', request.sid)

    @socketio.on('disconnect')
    def handle_disconnect():
        print('Client disconnected:', request.sid)

def emit_progress(socket, message, percentage):
    """
    Helper function to emit progress updates to the frontend.
    """
    socket.emit('progress', {'message': message, 'percentage': percentage})
    print({'message': message, 'percentage': percentage})