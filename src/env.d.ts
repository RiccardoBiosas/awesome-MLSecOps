/// <reference types="astro/client" />

interface ImportMetaEnv {
	readonly PUBLIC_GOOGLE_ANALYTICS_ID?: string;
	readonly PUBLIC_GOOGLE_SITE_VERIFICATION?: string;
}

interface ImportMeta {
	readonly env: ImportMetaEnv;
}

interface Window {
	dataLayer?: IArguments[];
	gtag?: (...args: unknown[]) => void;
	plausible?: ((eventName: string, options?: { props?: Record<string, string> }) => void) & {
		q?: IArguments[];
	};
}

declare module "@fontsource-variable/ibm-plex-sans";
declare module "@fontsource-variable/newsreader";