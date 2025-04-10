// API configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

export const API_ENDPOINTS = {
  LOAD_ISXD: `${API_BASE_URL}/load_isxd/`,
  GET_FRAME: `${API_BASE_URL}/get_frame/`,
  LIST_DIRECTORY: `${API_BASE_URL}/list_directory/`,
  LIST_ISXD_FILES: `${API_BASE_URL}/list_isxd_files/`,
  EXPORT_CROP_COORDS: `${API_BASE_URL}/export_crop_coords/`,
};
