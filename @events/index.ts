// Server and client will communicate on a duplex web-socket connection, 
// This file will define the spec for these events eg. image_push, can be a event triggered by the client when pushing a cropped frame for recognition.
// Similarly server_response can be an event triggered by the server when sending response, 
// ID ca be assosicated with one pair of these exchanges, this id can be attached to the end of the event name, eg server_response_[ID], 
// which will allow the client to asynchrnously listen for multiple responses. (eXtended Functionality)
