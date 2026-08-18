function martini2024_configure_batch()
%MARTINI2024_CONFIGURE_BATCH Hide figures when MARTINI2024_BATCH is set.
if ~isempty(getenv('MARTINI2024_BATCH'))
    set(0, 'DefaultFigureVisible', 'off');
end
end
