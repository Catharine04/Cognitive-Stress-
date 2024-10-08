import { useState, useEffect } from "react"
import SpotifyPlayer from "react-spotify-web-playback"

export default function MusicPlayerPlayer({ accessToken, trackUri }) {

  return (
    <SpotifyPlayer
    styles={{
      bgColor: 'black',            // Background color set to black
      color: '#fff',               // Text color set to white
      loaderColor: '#fff',
      sliderColor: '#1cb954',
      savedColor: '#fff',
      trackArtistColor: '#ccc',
      trackNameColor: '#fff',
    }}
      token={accessToken}
      uris={trackUri}
    />
  )
}