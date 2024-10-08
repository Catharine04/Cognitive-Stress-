import { useState } from "react";

function SearchBar() {
  const [inputValue, setInputValue] = useState("");
  const [tags, setTags] = useState([]);

  // Handle input change
  const handleInputChange = (e) => {
    setInputValue(e.target.value);
  };

  // Handle when user presses Enter
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && inputValue.trim()) {
      // Prevent adding duplicate tags
      if (!tags.includes(inputValue)) {
        setTags([...tags, inputValue]);
      }
      setInputValue(""); // Clear input field after adding
    }
  };

  // Remove a tag
  const removeTag = (index) => {
    setTags(tags.filter((_, i) => i !== index));
  };

  return (
    <div className="w-full max-w-lg">
      {/* Input Field */}
      <input
        type="text"
        placeholder="Search songs or artists"
        value={inputValue}
        onChange={handleInputChange}
        onKeyDown={handleKeyDown}
        className="bg-transparent border-2 border-white px-8 py-4 rounded-lg text-white text-lg font-bold transition duration-300 ease-in-out focus:outline-none focus:bg-white focus:text-black w-full mb-4"
      />

      {/* Display Tags */}
      <div className="flex flex-wrap gap-2">
        {tags.map((tag, index) => (
          <div
            key={index}
            className="bg-white text-black px-4 py-2 rounded-full flex items-center"
          >
            <span>{tag}</span>
            <button
              className="ml-2 text-black font-bold"
              onClick={() => removeTag(index)}
            >
              &times;
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

export default SearchBar;
