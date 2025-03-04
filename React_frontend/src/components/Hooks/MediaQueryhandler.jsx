import { useMediaQuery } from "react-responsive";

const MediaQueryHandler = () => {
  const isMobile = useMediaQuery({ maxWidth: 768 });
  return { isMobile };
};

export default MediaQueryHandler;
