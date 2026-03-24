def center_window(window, app_width, app_length):
    screen_width = window.winfo_screenwidth()
    screen_length = window.winfo_screenheight()
    x = int((screen_width/2) - (app_width / 2))
    y = int((screen_length/2) - (app_length/ 2))
    return window.geometry(f"{app_width}x{app_length}+{x}+{y}")
