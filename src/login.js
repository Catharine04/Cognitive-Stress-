// src/components/Login.js
import React from "react";

function login() {
  const CLIENT_ID = "0461e872c18c40278b9303528b68fbc5";
  const REDIRECT_URI = "http://localhost:3000";
  const AUTH_ENDPOINT = "https://accounts.spotify.com/authorize";
  const RESPONSE_TYPE = "token";
  const SCOPE = "user-top-read user-read-recently-played streaming user-read-email user-read-private user-read-playback-state user-modify-playback-state"

  return (
    <button  className="bg-transparent mt-8 hover:bg-white hover:text-black border-2 border-white px-8 py-4 rounded-lg text-white text-lg font-bold transition duration-300 ease-in-out">
    <a href={`${AUTH_ENDPOINT}?client_id=${CLIENT_ID}&redirect_uri=${REDIRECT_URI}&response_type=${RESPONSE_TYPE}&scope=${SCOPE}`}>Login to Spotify</a>
  </button>
  );
}

export default login;
