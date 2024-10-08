// src/components/CsvDownload.js
import React from "react";

function CsvDownload({ data }) {
  const appendToCSV = (newData) => {
    // Define the features to include in the CSV file
    const selectedFeatures = ["id", "danceability", "energy","key", "loudness","mode", "speechiness","acousticness", "instrumentalness",'liveness',"valence", "tempo"];
    
    // Generate CSV content with selected features
    let csvContent = "";
  
    // Check if existing CSV data is available in localStorage
    const existingCSVData = window.localStorage.getItem("music_features_csv");
  
    if (existingCSVData) {
      // Append new data to existing CSV data
      csvContent = existingCSVData + "\n";
    } else {
      // If no existing data, add headers
      csvContent = selectedFeatures.join(",") + "\n";
    }
  
    // Add new data rows
    newData.forEach(item => {
      const rowData = selectedFeatures.map(feature => item[feature] || "").join(",");
      csvContent += rowData + "\n";
    });
  
    // Save the updated CSV data to localStorage
    window.localStorage.setItem("music_features_csv", csvContent);
  
    // Convert CSV content to Blob
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  
    // Create a temporary URL for the Blob
    const url = URL.createObjectURL(blob);
  
    // Create an anchor element to trigger the download
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", "music_features.csv");
  
    // Append the anchor element to the document and trigger the click event
    document.body.appendChild(link);
    link.click();
  
    // Clean up by removing the anchor element and revoking the URL
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };
  

  return (
    <div>
      <button onClick={() => appendToCSV(data)}>csv</button>
    </div>
  );
}

export default CsvDownload;

