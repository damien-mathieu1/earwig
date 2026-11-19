//! Embedded metadata (ID3, Vorbis comments, MP4 atoms, RIFF INFO…) via lofty.

use std::path::Path;

use lofty::prelude::*;

#[derive(Debug, Default)]
pub struct FileTags {
    pub title: Option<String>,
    pub artist: Option<String>,
    pub album: Option<String>,
    pub genre: Option<String>,
    pub year: Option<i64>,
}

/// Missing or unreadable tags are not an error: the audio still gets analysed.
pub fn read(path: &Path) -> FileTags {
    let Ok(file) = lofty::read_from_path(path) else {
        return FileTags::default();
    };
    let Some(tag) = file.primary_tag().or_else(|| file.first_tag()) else {
        return FileTags::default();
    };
    let clean = |s: Option<std::borrow::Cow<'_, str>>| {
        s.map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
    };
    FileTags {
        title: clean(tag.title()),
        artist: clean(tag.artist()),
        album: clean(tag.album()),
        genre: clean(tag.genre()),
        year: tag.date().map(|d| d.year as i64),
    }
}
