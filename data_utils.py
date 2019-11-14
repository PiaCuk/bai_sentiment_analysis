def rpad(array, n=30):
    """Right padding to length n
    """
    current_len = len(array)
    if current_len > n:
        return array[: n - 1]
    extra = n - current_len
    return array + ([0] * extra)